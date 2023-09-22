import torch
import torch.nn as nn


class CorrTorch(nn.Module):
    def __init__(self, pad_size=1, kernel_size=1, max_displacement=1, stride1=1, stride2=1, corr_multiply=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad3d(pad_size, 0)

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsetz, offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1)])

        dep, hei, wid = in1.shape[2], in1.shape[3], in1.shape[4]
        output = torch.cat([
            torch.mean(in1 * in2_pad[:, :, dz:dz + dep, dy:dy + hei, dx:dx + wid], 1, keepdim=True)
            for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1))
        ], 1)
        return output
