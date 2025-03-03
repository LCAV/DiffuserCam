"""Modified from https://github.com/Waller-Lab/DiffuserCam-Tutorial/blob/master/GD.py"""

import numpy as np
import matplotlib.pyplot as plt
from lensless.utils.plot import plot_image
import inspect
import pathlib as plib


class GradientDescentUpdate:
    """Gradient descent update techniques."""

    VANILLA = "vanilla"
    NESTEROV = "nesterov"
    FISTA = "fista"

    @staticmethod
    def all_values():
        vals = []
        for i in inspect.getmembers(GradientDescentUpdate):
            # remove private and protected functions, and this function
            if not i[0].startswith("_") and not callable(i[1]):
                vals.append(i[1])
        return vals


def create_init_matrices(psf):
    """
    Create initialization matrices

    Parameters
    ----------
    psf :py:class:`~numpy.ndarray`
        2-D array of PSF.

    Returns
    -------

    """
    pixel_start = (np.max(psf) + np.min(psf)) / 2
    x = np.ones(psf.shape) * pixel_start

    init_shape = psf.shape
    padded_shape = [next_pow2(2 * n - 1) for n in init_shape]
    starti = (padded_shape[0] - init_shape[0]) // 2
    endi = starti + init_shape[0]
    startj = (padded_shape[1] // 2) - (init_shape[1] // 2)
    endj = startj + init_shape[1]
    hpad = np.zeros(padded_shape)
    hpad[starti:endi, startj:endj] = psf

    H = np.fft.fft2(hpad, norm="ortho")
    Hadj = np.conj(H)

    def crop(X):
        return X[starti:endi, startj:endj]

    def pad(v):
        vpad = np.zeros(padded_shape).astype(np.complex64)
        vpad[starti:endi, startj:endj] = v
        return vpad

    utils = [crop, pad]
    v = np.real(pad(x))

    return H, Hadj, v, utils


def next_pow2(n):
    return int(2 ** np.ceil(np.log2(n)))


def grad(Hadj, H, vk, b, crop, pad):
    Av = calcA(H, vk, crop)
    diff = Av - b
    return np.real(calcAHerm(Hadj, diff, pad))


def calcA(H, vk, crop):
    Vk = np.fft.fft2(vk, norm="ortho")
    return crop(np.fft.ifftshift(np.fft.ifft2(H * Vk, norm="ortho")))


def calcAHerm(Hadj, diff, pad):
    xpad = pad(diff)
    X = np.fft.fft2(xpad, norm="ortho")
    return np.fft.ifftshift(np.fft.ifft2(Hadj * X, norm="ortho"))


def gd_update(params, parent_var):
    # extract variables
    vk = params["vk"]
    H, Hadj, b, crop, pad, alpha, proj = parent_var

    # apply gradient step
    gradient = grad(Hadj, H, vk, b, crop, pad)
    vk -= alpha * gradient
    vk = proj(vk)

    # update variables
    params["vk"] = vk
    return params


def nesterov_update(params, parent_var):
    # extract variables
    vk = params["vk"]
    p = params["p"]
    mu = params["mu"]
    H, Hadj, b, crop, pad, alpha, proj = parent_var

    # apply gradient step
    p_prev = p
    gradient = grad(Hadj, H, vk, b, crop, pad)
    p = mu * p - alpha * gradient
    vk += -mu * p_prev + (1 + mu) * p
    vk = proj(vk)

    # update variables
    params["vk"] = vk
    params["p"] = p
    return params


def fista_update(params, parent_var):
    # extract variables
    vk = params["vk"]
    xk = params["xk"]
    tk = params["tk"]
    H, Hadj, b, crop, pad, alpha, proj = parent_var

    # apply gradient step
    x_k1 = xk
    gradient = grad(Hadj, H, vk, b, crop, pad)
    vk -= alpha * gradient
    xk = proj(vk)
    t_k1 = (1 + np.sqrt(1 + 4 * tk**2)) / 2
    vk = xk + (tk - 1) / t_k1 * (xk - x_k1)
    tk = t_k1

    # update variables
    params["vk"] = vk
    params["tk"] = tk
    params["xk"] = xk
    return params


def non_neg(xi):
    xi = np.maximum(xi, 0)
    return xi


def grad_descent(
    psf,
    data,
    n_iter=100,
    non_neg_constraint=True,
    update_method=GradientDescentUpdate.FISTA,
    disp_iter=10,
    plot_pause=0.2,
    save=False,
):
    """
    Gradient descent reconstruction approach.

    Parameters
    ----------
    psf :py:class:`~numpy.ndarray`
        2-D array of PSF.
    data :py:class:`~numpy.ndarray`
        2-D array of raw measurement data.
    n_iter : int, optional
        Number of iterations.
    non_neg_constraint : bool, optional
        Whether to apply non-negativity constraint.
    update_method : GradientDescentUpdate member
        Which gradient descent method to apply: vanilla, Nesterov, or FISTA.
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
    assert (
        update_method in GradientDescentUpdate.all_values()
    ), f"update method '{update_method}' not supported."

    assert len(psf.shape) == len(data.shape)
    is_rgb = True if len(psf.shape) == 3 else False
    if not is_rgb:
        psf = psf[:, :, np.newaxis]
        data = data[:, :, np.newaxis]

    if non_neg_constraint:
        proj = non_neg  # Enforce non-negativity at every gradient step.
    else:

        def proj(x):
            return x

    H = []
    Hadj = []
    v = []
    crop = []
    pad = []
    alpha = []
    params = []
    for c in range(psf.shape[2]):
        # precompute some matrices
        _H, _Hadj, _v, _utils = create_init_matrices(psf[:, :, c])
        H.append(_H)
        Hadj.append(_Hadj)
        v.append(_v)
        crop.append(_utils[0])
        pad.append(_utils[1])

        alpha.append(np.real(1.8 / (np.max(Hadj[c] * H[c]))))

        # gradient descent variables
        _params = {"vk": _v}
        if update_method is GradientDescentUpdate.NESTEROV:
            _params.update(
                {
                    "p": 0,
                    "mu": 0.9,
                }
            )
        elif update_method is GradientDescentUpdate.FISTA:
            _params.update(
                {
                    "tk": 1,
                    "xk": _v,
                }
            )
        params.append(_params)

    if disp_iter is not None:
        ax = plot_image(data)
    else:
        ax = None
        disp_iter = n_iter + 1

    for i in range(n_iter):
        for c in range(psf.shape[2]):
            parent_var = [H[c], Hadj[c], data[:, :, c], crop[c], pad[c], alpha[c], proj]
            if update_method is GradientDescentUpdate.VANILLA:
                params[c] = gd_update(params[c], parent_var)
            elif update_method is GradientDescentUpdate.NESTEROV:
                params[c] = nesterov_update(params[c], parent_var)
            elif update_method is GradientDescentUpdate.FISTA:
                params[c] = fista_update(params[c], parent_var)

            if (i + 1) % disp_iter == 0:
                if is_rgb:
                    image = np.zeros(psf.shape)
                    for c in range(3):
                        image[:, :, c] = proj(crop[c](params[c]["vk"]))
                else:
                    image = proj(crop[0](params[0]["vk"]))
                ax = plot_image(image, ax=ax)
                ax.set_title("Reconstruction after iteration {}".format(i + 1))
                if save:
                    plt.savefig(plib.Path(save) / f"{i + 1}_diffusercam.png")
                plt.draw()
                plt.pause(plot_pause)

    if is_rgb:
        final_im = np.zeros(psf.shape)
        for c in range(3):
            final_im[:, :, c] = proj(crop[c](params[c]["vk"]))
    else:
        final_im = proj(crop[0](params[0]["vk"]))
    ax = plot_image(final_im, ax=ax)
    ax.set_title("Final reconstruction after {} iterations".format(n_iter))
    if save:
        plt.savefig(plib.Path(save) / f"{n_iter}_diffusercam.png")

    return final_im, ax
