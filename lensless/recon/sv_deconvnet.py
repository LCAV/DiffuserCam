# #############################################################################
# sv_deconvnet.py
# ======================
# Authors :
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import numpy as np
import torch
from lensless.recon.trainable_recon import TrainableReconstructionAlgorithm


def compute_weight_matrices(psf, K):
    Nx, Ny = psf.shape[1:3]

    # get centers of KxK patches
    centers = []
    for i in range(K):
        for j in range(K):
            centers.append((int((i + 0.5) * Nx / K), int((j + 0.5) * Ny / K)))

    # compute weight matrices
    Y, X = np.meshgrid(np.arange(Ny), np.arange(Nx))
    eps = 1e-4
    weight_mat = []
    for center in centers:
        weight = ((X - center[0]) ** 2 + (Y - center[1]) ** 2 + eps) ** (-0.5)
        weight_mat.append(weight)

    # normalize weight matrices
    sum_weights = np.sum(weight_mat, axis=0)
    for i in range(K * K):
        weight_mat[i] /= sum_weights

    # check that sums to 1 at each pixel
    sum_mat = np.sum(weight_mat, axis=0)
    assert np.allclose(sum_mat, 1.0)

    return weight_mat


class SVDeconvNet(TrainableReconstructionAlgorithm):
    def __init__(self, psf, dtype=None, K=3, **kwargs):
        """
        Constructor for SVDeconvNet as proposed in PhoCoLens: https://phocolens.github.io/

        Parameters
        ----------
        psf : :py:class:`~torch.Tensor`
            Point spread function (PSF) that models forward propagation.
            Must be of shape (depth, height, width, channels) even if
            depth = 1 and channels = 1. You can use :py:func:`~lensless.io.load_psf`
            to load a PSF from a file such that it is in the correct format.
        dtype : float32 or float64
            Data type to use for optimization.
        K : int
            (K x K) kernels are learned for spatially-variant deconvolution.

        """
        multipsf = psf.repeat(K * K, 1, 1, 1)

        # compute weight matrices
        self.weight_mat = torch.tensor(compute_weight_matrices(psf, K)).to(
            dtype=psf.dtype, device=psf.device
        )
        self.weight_mat = self.weight_mat[None, :, :, :, None]  # (batch, K*K, Nx, Ny, channels)
        super(SVDeconvNet, self).__init__(multipsf, n_iter=1, dtype=dtype, reset=False, **kwargs)
        self.reset()

    def _form_image(self):
        self._image_est[self._image_est < 0] = 0
        return self._image_est

    def _set_psf(self, psf):
        super()._set_psf(psf)

    def reset(self, batch_size=1):
        # no state variables
        return

    def _update(self, iter):
        self._image_est = torch.sum(
            self.weight_mat * self._convolver.deconvolve(self._data), dim=1, keepdim=True
        )
