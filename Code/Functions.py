import glob
import os

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_flow(inputs, times, target_flow=None, mode="trilinear"):
    if target_flow is not None:
        _, _, d, h, w = target_flow.size()
    else:
        raise ValueError('wrong input')
    _, _, d_, h_, w_ = inputs.size()
    res = F.interpolate(inputs, [d, h, w], mode=mode)
    res /= times
    return res


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z - 1)
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y - 1)
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x - 1)

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z - 1)
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y - 1)
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x - 1)

    return flow


def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    # X = np.reshape(X, (1,1,) + X.shape)
    X = np.reshape(X, (1,) + X.shape)
    return X


def crop_center(img, cropx, cropy, cropz):
    x, y, z = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    startz = z // 2 - cropz // 2
    return img[startx:startx + cropx, starty:starty + cropy, startz:startz + cropz]


def load_4D_with_crop(name, cropx, cropy, cropz):
    X = nib.load(name)
    X = X.get_fdata()

    x, y, z = X.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    startz = z // 2 - cropz // 2

    X = X[startx:startx + cropx, starty:starty + cropy, startz:startz + cropz]

    X = np.reshape(X, (1,) + X.shape)
    return X


def load_4D_with_header(name):
    X = nib.load(name)
    X_npy = X.get_fdata()
    X_npy = np.reshape(X_npy, (1,) + X_npy.shape)
    return X_npy, X.header, X.affine


def load_3D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    return X


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(img):
    i_max = np.max(img)
    i_min = np.min(img)
    norm = (img - i_min) / (i_max - i_min)
    return norm


def save_img(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_dir):
        'Initialization'
        super(Dataset_epoch, self).__init__()
        self.img_dir = img_dir
        self.files = sorted(glob.glob(os.path.join(self.img_dir, '*.nii.gz')))
        self.index_pair = list(itertools.permutations(self.files, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])
        return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Predict_dataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_dir, label_dir):
        'Initialization'
        super(Predict_dataset, self).__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.files = sorted(glob.glob(os.path.join(self.img_dir, '*.nii.gz')))
        self.index_pair = list(itertools.permutations(self.files, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A_name = self.index_pair[step][0]
        img_B_name = self.index_pair[step][1]
        label_A_name = img_A_name.split("/")[-1] + "-label.nii.gz"
        label_B_name = img_B_name.split("/")[-1] + "-label.nii.gz"
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])
        label_A = load_4D(os.path.join(self.label_dir, label_A_name))
        label_B = load_4D(os.path.join(self.label_dir, label_B_name))
        return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(
            label_A).float(), torch.from_numpy(label_B).float()


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', align_corners=True)

        return flow


class SpatialTransformNearest(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', align_corners=True)

        return flow


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid):
        flow = velocity / (2.0 ** self.time_step)
        size_tensor = sample_grid.size()
        for _ in range(self.time_step):
            grid = sample_grid + flow.permute(0, 2, 3, 4, 1)
            grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
            grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
            grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
        return flow


class CompositionTransform(nn.Module):
    def __init__(self):
        super(CompositionTransform, self).__init__()

    def forward(self, flow_1, flow_2, sample_grid):
        size_tensor = sample_grid.size()
        grid = sample_grid + flow_2.permute(0, 2, 3, 4, 1)
        grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
        grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
        grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
        compos_flow = F.grid_sample(flow_1, grid, mode='bilinear', align_corners=True) + flow_2
        return compos_flow
