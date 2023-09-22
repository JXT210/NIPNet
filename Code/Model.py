import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Functions import generate_grid, SpatialTransform, DiffeomorphicTransform
from CorrTorch import CorrTorch
from DFEM import DFEM


class Encoder(nn.Module):
    def __init__(self, dim, bn=True):
        super(Encoder, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = [8, 16, 16, 32, 32]
        # Encoder functions
        self.encoder1 = conv_block(dim, 1, self.enc_nf[0], stride=1, batchnorm=bn)  # 1

        self.encoder2 = conv_block(dim, self.enc_nf[0], self.enc_nf[1], stride=2, batchnorm=bn)  # 1 / 2
        self.DFEM2 = DFEM(self.enc_nf[1])

        self.encoder3 = conv_block(dim, self.enc_nf[1], self.enc_nf[2], stride=2, batchnorm=bn)  # 1 / 4
        self.DFEM3 = DFEM(self.enc_nf[2])

        self.encoder4 = conv_block(dim, self.enc_nf[2], self.enc_nf[3], stride=2, batchnorm=bn)  # 1 / 8
        self.DFEM4 = DFEM(self.enc_nf[3])

        self.encoder5 = conv_block(dim, self.enc_nf[3], self.enc_nf[4], stride=2, batchnorm=bn)  # 1 / 16
        self.DFEM5 = DFEM(self.enc_nf[4])

    def forward(self, x):
        # Get encoder activations
        x_enc = []
        x = self.encoder1(x)
        x_enc.append(x)
        x = self.DFEM2(self.encoder2(x))
        x_enc.append(x)
        x = self.DFEM3(self.encoder3(x))
        x_enc.append(x)
        x = self.DFEM4(self.encoder4(x))
        x_enc.append(x)
        x = self.DFEM5(self.encoder5(x))
        x_enc.append(x)
        return x, x_enc


def conv_block(dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
    conv_fn = getattr(nn, "Conv{0}d".format(dim))
    bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
    if batchnorm:
        layer = nn.Sequential(
            conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            bn_fn(out_channels),
            nn.LeakyReLU(0.2))
    else:
        layer = nn.Sequential(
            conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2))
    return layer


class DeformerLayer(nn.Module):
    def __init__(self, dim, input_channels):
        super(DeformerLayer, self).__init__()
        self.input_channels = input_channels
        self.corr = CorrTorch()
        self.conv = conv_block(dim, input_channels * 2 + 27, input_channels)

    def forward(self, fixed_feature, moving_feature):
        cost_volume = self.corr(fixed_feature, moving_feature)
        x = torch.cat((fixed_feature, cost_volume, moving_feature), dim=1)
        x = self.conv(x)
        return x


class Spacial_attention0(nn.Module):
    def __init__(self, in_channels, lay_channels):
        super(Spacial_attention0, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, lay_channels, 3, 1, 1, bias=False),
            nn.Conv3d(lay_channels, 2, 3, 1, 1, bias=False)
        )
        self.act = nn.Sigmoid()

    def forward(self, x0, x1):
        x = torch.cat((x0, x1), dim=1)
        out = self.conv(x)
        weight_map = self.act(out)
        return torch.cat((x0 * torch.unsqueeze(weight_map[:, 0, :, :, :], dim=1),
                          x1 * torch.unsqueeze(weight_map[:, 1, :, :, :], dim=1)), dim=1)


class Spacial_attention1(nn.Module):
    def __init__(self, in_channels, lay_channels):
        super(Spacial_attention1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, lay_channels, 3, 1, 1, bias=False),
            nn.Conv3d(lay_channels, 3, 3, 1, 1, bias=False)
        )
        self.act = nn.Sigmoid()

    def forward(self, x0, x1, x2):
        x = torch.cat((x0, x1, x2), dim=1)
        out = self.conv(x)
        weight_map = self.act(out)
        return torch.cat((x0 * torch.unsqueeze(weight_map[:, 0, :, :, :], dim=1),
                          x1 * torch.unsqueeze(weight_map[:, 1, :, :, :], dim=1),
                          x2 * torch.unsqueeze(weight_map[:, 2, :, :, :], dim=1)), dim=1)


class NIPNet(nn.Module):
    def __init__(self, dim, device, imgshape):
        super(NIPNet, self).__init__()
        # One conv to get the feature map
        self.backbone = Encoder(dim)
        self.enc_nf = [8, 16, 16, 32, 32]
        self.device = device
        self.imgshape1 = imgshape
        self.imgshape2 = tuple([int(x / 2) for x in imgshape])
        self.imgshape4 = tuple([int(x / 4) for x in imgshape])
        self.imgshape8 = tuple([int(x / 8) for x in imgshape])
        self.imgshape16 = tuple([int(x / 16) for x in imgshape])

        self.grid_1 = generate_grid(self.imgshape1)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).to(device).float()

        self.grid_2 = generate_grid(self.imgshape2)
        self.grid_2 = torch.from_numpy(np.reshape(self.grid_2, (1,) + self.grid_2.shape)).to(device).float()

        self.grid_4 = generate_grid(self.imgshape4)
        self.grid_4 = torch.from_numpy(np.reshape(self.grid_4, (1,) + self.grid_4.shape)).to(device).float()

        self.grid_8 = generate_grid(self.imgshape8)
        self.grid_8 = torch.from_numpy(np.reshape(self.grid_8, (1,) + self.grid_8.shape)).to(device).float()

        self.grid_16 = generate_grid(self.imgshape16)
        self.grid_16 = torch.from_numpy(np.reshape(self.grid_16, (1,) + self.grid_16.shape)).to(device).float()

        self.transform = SpatialTransform().to(device)

        # NIPNet-diff version
        # self.diff_transform = DiffeomorphicTransform().to(device)

        self.up = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.deformer_lay1_feature = DeformerLayer(dim, self.enc_nf[4])  # 1 / 16
        self.deformer_lay1_flow = nn.Sequential(
            self.outputs(self.enc_nf[4] + self.enc_nf[3], 3)
        )
        self.CBAM_lay1 = Spacial_attention0(self.enc_nf[4] + self.enc_nf[3], self.enc_nf[4])

        self.deformer_lay2_feature = DeformerLayer(dim, self.enc_nf[3])  # 1 / 8
        self.deformer_lay2_flow = nn.Sequential(
            self.outputs(self.enc_nf[4] + self.enc_nf[3] + self.enc_nf[2], 3)
        )
        self.CBAM_lay2 = Spacial_attention1(self.enc_nf[4] + self.enc_nf[3] + self.enc_nf[2], self.enc_nf[3])

        self.deformer_lay3_feature = DeformerLayer(dim, self.enc_nf[2])  # 1 / 4
        self.deformer_lay3_flow = nn.Sequential(
            self.outputs(self.enc_nf[3] + self.enc_nf[2] + self.enc_nf[1], 3)
        )
        self.CBAM_lay3 = Spacial_attention1(self.enc_nf[3] + self.enc_nf[2] + self.enc_nf[1], self.enc_nf[2])
        self.deformer_lay4_feature = DeformerLayer(dim, self.enc_nf[1])  # 1  / 2
        self.deformer_lay4_flow = nn.Sequential(
            self.outputs(self.enc_nf[2] + self.enc_nf[1] + self.enc_nf[0], 3)
        )
        self.CBAM_lay4 = Spacial_attention1(self.enc_nf[2] + self.enc_nf[1] + self.enc_nf[0], self.enc_nf[1])
        self.deformer_lay5_feature = DeformerLayer(dim, self.enc_nf[0])  # 1
        self.deformer_lay5_flow = nn.Sequential(
            self.outputs(self.enc_nf[1] + self.enc_nf[0], 3)
        )
        self.CBAM_lay5 = Spacial_attention0(self.enc_nf[1] + self.enc_nf[0], self.enc_nf[0])

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, in_channels // 2, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(in_channels // 2, 3, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def asff0(self, fixed_feature1, moving_feature1, fixed_feature2, moving_feature2,
              deformer_lay1, deformer_lay2, deformer_flow, CBAM_lay):

        feature1 = deformer_lay1(fixed_feature1, moving_feature1)
        feature2 = deformer_lay2(fixed_feature2, moving_feature2)
        feature2 = F.interpolate(feature2, scale_factor=0.5, mode="trilinear", align_corners=True)
        feature = CBAM_lay(feature1, feature2)
        flow = deformer_flow(feature)
        return flow

    def asff1(self, fixed_feature0, moving_feature0, fixed_feature1, moving_feature1, fixed_feature2, moving_feature2,
              deformer_lay0, deformer_lay1, deformer_lay2, deformer_flow, CBAM_lay, transform=None, grid0=None,
              grid1=None, grid2=None, flow0=None, flow=None):
        if flow is not None:
            moving_feature0 = transform(moving_feature0, flow0.permute(0, 2, 3, 4, 1), grid0)
            moving_feature1 = transform(moving_feature1, flow.permute(0, 2, 3, 4, 1), grid1)
            flow2 = self.up(flow * 2)
            moving_feature2 = transform(moving_feature2, flow2.permute(0, 2, 3, 4, 1), grid2)
        feature0 = deformer_lay0(fixed_feature0, moving_feature0)
        feature0 = F.interpolate(feature0, scale_factor=2, mode="trilinear", align_corners=True)
        feature1 = deformer_lay1(fixed_feature1, moving_feature1)
        feature2 = deformer_lay2(fixed_feature2, moving_feature2)
        feature2 = F.interpolate(feature2, scale_factor=0.5, mode="trilinear", align_corners=True)
        feature = CBAM_lay(feature0, feature1, feature2)
        flow = deformer_flow(feature)
        return flow

    def asff2(self, fixed_feature0, moving_feature0, fixed_feature1, moving_feature1,
              deformer_lay0, deformer_lay1, deformer_flow, CBAM_lay, transform=None, grid0=None,
              grid1=None, flow0=None, flow=None):
        if flow is not None:
            moving_feature0 = transform(moving_feature0, flow0.permute(0, 2, 3, 4, 1), grid0)
            moving_feature1 = transform(moving_feature1, flow.permute(0, 2, 3, 4, 1), grid1)
        feature0 = deformer_lay0(fixed_feature0, moving_feature0)
        feature0 = F.interpolate(feature0, scale_factor=2, mode="trilinear", align_corners=True)
        feature1 = deformer_lay1(fixed_feature1, moving_feature1)

        feature = CBAM_lay(feature0, feature1)
        flow = deformer_flow(feature)
        return flow

    def forward(self, fixed, moving):
        # batch channel depth width height
        input = torch.cat([moving, fixed], dim=0)
        _, features = self.backbone(input)
        features.reverse()
        b = moving.shape[0]
        moving_feature = []
        fixed_feature = []
        for i in features:
            moving_feature.append(i[0:b])
            fixed_feature.append(i[b:])
        intermediate_flows = []
        # first level
        flow_lay1 = self.asff0(fixed_feature[0], moving_feature[0], fixed_feature[1], moving_feature[1],
                               self.deformer_lay1_feature, self.deformer_lay2_feature, self.deformer_lay1_flow,
                               self.CBAM_lay1)  # 1 / 16
        intermediate_flows.append(flow_lay1)
        flow1 = self.up(flow_lay1 * 2)
        # second level
        flow_lay2 = self.asff1(fixed_feature[0], moving_feature[0], fixed_feature[1], moving_feature[1],
                               fixed_feature[2], moving_feature[2], self.deformer_lay1_feature,
                               self.deformer_lay2_feature, self.deformer_lay3_feature, self.deformer_lay2_flow,
                               self.CBAM_lay2, self.transform, self.grid_16, self.grid_8, self.grid_4, flow_lay1,
                               flow1)  # 1 / 8
        flow_lay2 = flow_lay2 + flow1
        intermediate_flows.append(flow_lay2)
        flow2 = self.up(flow_lay2 * 2)

        # third level
        flow_lay3 = self.asff1(fixed_feature[1], moving_feature[1], fixed_feature[2], moving_feature[2],
                               fixed_feature[3], moving_feature[3], self.deformer_lay2_feature,
                               self.deformer_lay3_feature, self.deformer_lay4_feature, self.deformer_lay3_flow,
                               self.CBAM_lay3, self.transform, self.grid_8, self.grid_4, self.grid_2, flow_lay2,
                               flow2)  # 1 / 4
        flow_lay3 = flow_lay3 + flow2
        intermediate_flows.append(flow_lay3)
        flow3 = self.up(flow_lay3 * 2)

        # fourth level
        flow_lay4 = self.asff1(fixed_feature[2], moving_feature[2], fixed_feature[3], moving_feature[3],
                               fixed_feature[4], moving_feature[4], self.deformer_lay3_feature,
                               self.deformer_lay4_feature, self.deformer_lay5_feature, self.deformer_lay4_flow,
                               self.CBAM_lay4, self.transform, self.grid_4, self.grid_2, self.grid_1, flow_lay3,
                               flow3)  # 1 / 2
        flow_lay4 = flow_lay4 + flow3
        intermediate_flows.append(flow_lay4)
        flow4 = self.up(flow_lay4 * 2)

        # final level
        flow_lay5 = self.asff2(fixed_feature[3], moving_feature[3], fixed_feature[4], moving_feature[4],
                               self.deformer_lay4_feature, self.deformer_lay5_feature, self.deformer_lay5_flow,
                               self.CBAM_lay5, self.transform, self.grid_2, self.grid_1, flow_lay4, flow4)
        flow = flow_lay5 + flow4
        # flow = self.diff_transform(flow, self.grid_1)
        return flow, intermediate_flows


if __name__ == "__main__":
    device = torch.device('cuda')
    x1 = torch.randn([1, 1, 160, 192, 224]).to(device)
    x2 = torch.randn([1, 1, 160, 192, 224]).to(device)
    model = NIPNet(3, device, imgshape=(160, 192, 224)).to(device)
    flow, intermediate_flows = model(x1, x2)
    print(flow.shape)
