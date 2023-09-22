import torch
import torch.fft
import torch.nn.functional as F
import torch.nn as nn


def conv_block(dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=True):
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


class DFEM(nn.Module):
    def __init__(self, input_channels, a=10, b=12, c=10):
        super(DFEM, self).__init__()
        self.register_buffer("input_channels", torch.as_tensor(input_channels))
        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))
        self.register_buffer("c", torch.as_tensor(c))

        # channels depth height width
        self.a_weight = nn.Parameter(torch.Tensor(2, input_channels // 2, a, b, c))
        nn.init.ones_(self.a_weight)

        self.conv = conv_block(dim=3, in_channels=input_channels // 2, out_channels=input_channels // 2)

        self.wg_a = conv_block(dim=3, in_channels=input_channels // 2, out_channels=input_channels // 2)

        self.project = conv_block(dim=3, in_channels=input_channels, out_channels=input_channels)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        # batch_size channels depth height wdith
        B, c, d, h, w = x1.size()
        # ----- FFT -----#
        x1 = torch.fft.rfftn(x1, dim=(2, 3, 4), norm='ortho')  # B, c, d, h, w
        a_weight = self.a_weight
        a_weight = self.wg_a(F.interpolate(a_weight, size=x1.shape[2:5],
                                           mode='trilinear', align_corners=True)).permute(1, 2, 3, 4, 0)
        a_weight = torch.view_as_complex(a_weight.contiguous())
        x1 = x1 * a_weight
        x1 = torch.fft.irfftn(x1, s=(d, h, w), dim=(2, 3, 4), norm='ortho')

        # ----- convlution -----#
        x2 = self.conv(x2)

        # ----- concat -----#
        out = torch.cat([x1, x2], dim=1)
        out = out + x
        out = self.project(out)
        return out
