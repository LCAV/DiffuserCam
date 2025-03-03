"""Modified from https://github.com/Waller-Lab/DiffuserCam-Tutorial/blob/master/ADMM.py"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib as plib
from lensless.utils.plot import plot_image


def U_update(eta, image_est, param):
    """Total variation update."""
    return soft_thresh(Psi(image_est) + eta / param["mu2"], param["tau"] / param["mu2"])


def soft_thresh(x, thresh):
    # numpy automatically applies functions to each element of the array
    return np.sign(x) * np.maximum(0, np.abs(x) - thresh)


def Psi(v):
    """Gradient of image estimate, approximated by finite difference."""
    return np.stack((np.roll(v, 1, axis=0) - v, np.roll(v, 1, axis=1) - v), axis=2)


def PsiT(U):
    """Adjoint of of finite difference, corresponds to sum of two background differences."""
    diff1 = np.roll(U[..., 0], -1, axis=0) - U[..., 0]
    diff2 = np.roll(U[..., 1], -1, axis=1) - U[..., 1]
    return diff1 + diff2


def X_update(xi, image_est, H_fft, sensor_reading, X_divmat, param):
    return X_divmat * (xi + param["mu1"] * conv(image_est, H_fft) + crop_adj(sensor_reading, param))


def conv(vk, H_fft):
    """Convolution operator."""
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(vk)) * H_fft)))


def conv_adj(x, H_fft):
    """Adjoint of convolution operator."""
    x_zeroed = np.fft.ifftshift(x)
    return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(x_zeroed) * np.conj(H_fft))))


def crop(sensor_reading, param):
    """Crop operation."""
    # Image stored as matrix (row-column rather than x-y)
    top = (param["full_size"][0] - param["sensor_size"][0]) // 2
    bottom = (param["full_size"][0] + param["sensor_size"][0]) // 2
    left = (param["full_size"][1] - param["sensor_size"][1]) // 2
    right = (param["full_size"][1] + param["sensor_size"][1]) // 2
    return sensor_reading[top:bottom, left:right]


def crop_adj(sensor_reading, param):
    """Adjoint of crop, namely zero-padding."""

    v_pad = (param["full_size"][0] - param["sensor_size"][0]) // 2
    h_pad = (param["full_size"][1] - param["sensor_size"][1]) // 2
    return np.pad(
        sensor_reading, ((v_pad, v_pad), (h_pad, h_pad)), "constant", constant_values=(0, 0)
    )


def precompute_X_divmat(param):
    """Only call this function once!
    Store it in a variable and only use that variable
    during every update step"""
    return 1.0 / (crop_adj(np.ones(param["sensor_size"]), param) + param["mu1"])


def W_update(rho, image_est, param):
    """Non-negativity update"""
    return np.maximum(rho / param["mu3"] + image_est, 0)


def r_calc(w, rho, u, eta, x, xi, H_fft, param):
    return (
        (param["mu3"] * w - rho)
        + PsiT(param["mu2"] * u - eta)
        + conv_adj(param["mu1"] * x - xi, H_fft)
    )


def V_update(w, rho, u, eta, x, xi, H_fft, R_divmat, param):
    """Image update estimate."""
    freq_space_result = R_divmat * np.fft.fft2(
        np.fft.ifftshift(r_calc(w, rho, u, eta, x, xi, H_fft, param))
    )
    return np.real(np.fft.fftshift(np.fft.ifft2(freq_space_result)))


def precompute_PsiTPsi(param):
    PsiTPsi = np.zeros(param["full_size"])
    PsiTPsi[0, 0] = 4
    PsiTPsi[0, 1] = PsiTPsi[1, 0] = PsiTPsi[0, -1] = PsiTPsi[-1, 0] = -1
    PsiTPsi = np.fft.fft2(PsiTPsi)
    return PsiTPsi


def precompute_R_divmat(H_fft, PsiTPsi, param):
    """Only call this function once!
    Store it in a variable and only use that variable
    during every update step"""
    MTM_component = param["mu1"] * (np.abs(np.conj(H_fft) * H_fft))
    PsiTPsi_component = param["mu2"] * np.abs(PsiTPsi)
    id_component = param["mu3"]
    """This matrix is a mask in frequency space. So we will only use
    it on images that have already been transformed via an fft"""
    return 1.0 / (MTM_component + PsiTPsi_component + id_component)


def xi_update(xi, V, H_fft, X, param):
    return xi + param["mu1"] * (conv(V, H_fft) - X)


def eta_update(eta, V, U, param):
    return eta + param["mu2"] * (Psi(V) - U)


def rho_update(rho, V, W, param):
    return rho + param["mu3"] * (V - W)


def init_matrices(H_fft, param):
    X = np.zeros(param["full_size"])
    U = np.zeros((param["full_size"][0], param["full_size"][1], 2))
    V = np.zeros(param["full_size"])
    W = np.zeros(param["full_size"])

    xi = np.zeros_like(conv(V, H_fft))
    eta = np.zeros_like(Psi(V))
    rho = np.zeros_like(W)
    return X, U, V, W, xi, eta, rho


def precompute_H_fft(psf, param):
    return np.fft.fft2(np.fft.ifftshift(crop_adj(psf, param)))


def admm_step(X, U, V, W, xi, eta, rho, precomputed, param):
    H_fft, data, X_divmat, R_divmat = precomputed
    U = U_update(eta, V, param)  # TODO : pass U or V?
    X = X_update(xi, V, H_fft, data, X_divmat, param)  # TODO : pass X or V?
    V = V_update(W, rho, U, eta, X, xi, H_fft, R_divmat, param)
    W = W_update(rho, V, param)
    xi = xi_update(xi, V, H_fft, X, param)
    eta = eta_update(eta, V, U, param)
    rho = rho_update(rho, V, W, param)

    return X, U, V, W, xi, eta, rho


def apply_admm(psf, data, param, n_iter=5, disp_iter=1, plot_pause=0.5, plot=True, save=False):
    """
    ADMM reconstruction approach.

    Parameters
    ----------
    psf :py:class:`~numpy.ndarray`
        2-D array of PSF.
    data :py:class:`~numpy.ndarray`
        2-D array of raw measurement data.
    param : dict
        ADMM parameters
    n_iter : int, optional
        Number of iterations.
    disp_iter : int, optional
        How many iterations to wait before plotting. Set to None for no intermediate plots.
    plot_pause : float, optional
        How much to pause (in seconds) between iterations plot.

    Returns
    -------
    final_im :py:class:`~numpy.ndarray`
        2-D array of reconstructed data.
    ax : :py:class:`~matplotlib.axes.Axes`
    """
    assert "mu1" in param.keys()
    assert "mu2" in param.keys()
    assert "mu3" in param.keys()
    assert "tau" in param.keys()
    assert "sensor_size" in param.keys()
    assert "full_size" in param.keys()

    assert len(psf.shape) == len(data.shape)
    is_rgb = True if len(psf.shape) == 3 else False
    if not is_rgb:
        psf = psf[:, :, np.newaxis]
        data = data[:, :, np.newaxis]

    # precompute some matrices
    H_fft = []
    X = []
    U = []
    V = []
    W = []
    xi = []
    eta = []
    rho = []
    X_divmat = []
    PsiTPsi = []
    R_divmat = []
    for c in range(psf.shape[2]):
        H_fft.append(precompute_H_fft(psf[:, :, c], param))
        _X, _U, _V, _W, _xi, _eta, _rho = init_matrices(H_fft[c], param)
        X.append(_X)
        U.append(_U)
        V.append(_V)
        W.append(_W)
        xi.append(_xi)
        eta.append(_eta)
        rho.append(_rho)
        X_divmat.append(precompute_X_divmat(param))
        PsiTPsi.append(precompute_PsiTPsi(param))
        R_divmat.append(precompute_R_divmat(H_fft[c], PsiTPsi[c], param))

    if plot and disp_iter is not None:
        if is_rgb:
            ax = plot_image(data)
        else:
            ax = plot_image(data[:, :, 0])
    else:
        ax = None
        disp_iter = n_iter + 1

    for i in range(n_iter):
        for c in range(psf.shape[2]):
            X[c], U[c], V[c], W[c], xi[c], eta[c], rho[c] = admm_step(
                X[c],
                U[c],
                V[c],
                W[c],
                xi[c],
                eta[c],
                rho[c],
                precomputed=[H_fft[c], data[:, :, c], X_divmat[c], R_divmat[c]],
                param=param,
            )

        if plot and (i + 1) % disp_iter == 0:
            if is_rgb:
                image = np.zeros(tuple(param["sensor_size"]) + (3,))
                for c in range(3):
                    image[:, :, c] = crop(V[c], param)
            else:
                image = crop(V[0], param)
            image[image < 0] = 0
            ax = plot_image(image, ax=ax)
            ax.set_title("Reconstruction after iteration {}".format(i + 1))
            if save:
                plt.savefig(plib.Path(save) / f"{i + 1}_diffusercam.png")
            plt.draw()
            plt.pause(plot_pause)

    if is_rgb:
        final_im = np.zeros(tuple(param["sensor_size"]) + (3,))
        for c in range(3):
            final_im[:, :, c] = crop(V[c], param)
    else:
        final_im = crop(V[0], param)
    final_im[final_im < 0] = 0
    if plot:
        ax = plot_image(final_im, ax=ax)
        ax.set_title("Final reconstruction after {} iterations".format(n_iter))
        if save:
            plt.savefig(plib.Path(save) / f"{n_iter}_diffusercam.png")
        return final_im, ax
    else:
        return final_im
