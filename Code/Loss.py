import torch
import numpy as np
import torch.nn.functional as F


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def down_sample_flow(input_flow, target_flow, mode="trilinear"):
    # input_flow is final_flow
    # target_flow is intermediate_flow
    # so input_flow >> target_flow
    _, _, d, h, w = target_flow.size()
    _, _, d_, h_, w_ = input_flow.size()
    scale = d_ / d
    res = F.interpolate(input_flow, [d, h, w], mode=mode, align_corners=True)
    res /= scale
    return res


def PDLoss(final_flow, intermediate_flows):
    total_loss = 0
    for i in range(len(intermediate_flows)):
        flow = down_sample_flow(final_flow, intermediate_flows[i])
        d_loss = mse_loss(flow, intermediate_flows[i])
        total_loss += d_loss
    return total_loss


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# """
# Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
# """
#
#
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=5, eps=1e-8):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device,
                            requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)
        # return negative cc.
        return -1.0 * torch.mean(cc)


def compute_label_dice(gt, pred, dataset="OASIS"):
    '''The label category to be calculated, excluding background and non-existent areas in the image'''
    if dataset == 'OASIS':
        cls_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35]
    elif dataset == 'LPBA40':
        cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61,
                   62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161,
                   162,
                   163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)
