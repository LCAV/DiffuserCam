from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.metrics import structural_similarity

import torch


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def l2(p0, p1, range=255.0):
    return 0.5 * np.mean((p0 / range - p1 / range) ** 2)


def psnr(p0, p1, peak=255.0):
    return 10 * np.log10(peak**2 / np.mean((1.0 * p0 - 1.0 * p1) ** 2))


def dssim(p0, p1, range=255.0):
    return (1 - structural_similarity(p0, p1, data_range=range, multichannel=True)) / 2.0


def tensor2np(tensor_obj):
    # change dimension of a tensor object into a numpy array
    return tensor_obj[0].cpu().float().numpy().transpose((1, 2, 0))


def np2tensor(np_obj):
    # change dimenion of np array into tensor array
    return torch.Tensor(np_obj[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    # image tensor to lab tensor
    from skimage import color

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if to_norm and not mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.0

    return np2tensor(img_lab)


def tensorlab2tensor(lab_tensor, return_inbnd=False):
    from skimage import color
    import warnings

    warnings.filterwarnings("ignore")

    lab = tensor2np(lab_tensor) * 100.0
    lab[:, :, 0] = lab[:, :, 0] + 50

    rgb_back = 255.0 * np.clip(color.lab2rgb(lab.astype("float")), 0, 1)
    if return_inbnd:
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype("uint8"))
        mask = 1.0 * np.isclose(lab_back, lab, atol=2.0)
        mask = np2tensor(np.prod(mask, axis=2)[:, :, np.newaxis])
        return (im2tensor(rgb_back), mask)
    else:
        return im2tensor(rgb_back)


def tensor2im(image_tensor, imtype=np.uint8, cent=1.0, factor=255.0 / 2.0):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


def im2tensor(image, imtype=np.uint8, cent=1.0, factor=255.0 / 2.0):
    return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def tensor2vec(vector_tensor):
    return vector_tensor.data.cpu().numpy()[:, :, 0, 0]


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
