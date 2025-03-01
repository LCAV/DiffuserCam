import pytest
from lensless.utils.io import load_data
import numpy as np

try:
    import pycsou
    from lensless import GradientDescent, NesterovGradientDescent, FISTA, ADMM, APGD, APGDPriors

    pycsou_available = True

except ImportError:
    from lensless import GradientDescent, NesterovGradientDescent, FISTA, ADMM

    pycsou_available = False


try:
    import torch
    from lensless import UnrolledFISTA, UnrolledADMM

    torch_is_available = True
    torch.autograd.set_detect_anomaly(True)
    trainable_algos = [UnrolledFISTA, UnrolledADMM]
except ImportError:
    torch_is_available = False
    trainable_algos = []


psf_fp = "data/psf/tape_rgb.png"
data_fp = "data/raw_data/thumbs_up_rgb.png"
dtypes = ["float32", "float64"]
downsample = 16
_n_iter = 5

# classical algorithms
standard_algos = [GradientDescent, NesterovGradientDescent, FISTA, ADMM]


@pytest.mark.parametrize("algorithm", standard_algos)
def test_set_initial_est(algorithm):
    for gray in [True, False]:
        psf, _ = load_data(
            psf_fp=psf_fp,
            data_fp=data_fp,
            downsample=downsample,
            plot=False,
            gray=gray,
            use_torch=False,
        )
        recon = algorithm(psf)
        assert recon._initial_est is None
        random_init = np.random.rand(*recon._image_est_shape)
        recon._set_initial_estimate(random_init)
        assert isinstance(recon._initial_est, np.ndarray)
        assert np.allclose(recon._initial_est, random_init)

        # # set from constructor
        recon = algorithm(psf, initial_est=random_init)
        assert isinstance(recon._initial_est, np.ndarray)
        assert np.allclose(recon._initial_est, random_init)


@pytest.mark.parametrize("algorithm", trainable_algos)
def test_set_initial_est_unrolled(algorithm):
    if not torch_is_available:
        return
    for gray in [True, False]:
        psf, _ = load_data(
            psf_fp=psf_fp,
            data_fp=data_fp,
            downsample=downsample,
            plot=False,
            gray=gray,
            use_torch=True,
        )
        recon = algorithm(psf)
        assert recon._initial_est is None
        random_init = torch.rand(*recon._image_est.shape)
        recon._set_initial_estimate(random_init)
        assert isinstance(recon._initial_est, torch.Tensor)
        assert np.allclose(recon._initial_est, random_init)

        # set from constructor
        recon = algorithm(psf, initial_est=random_init)
        assert isinstance(recon._initial_est, torch.Tensor)
        assert np.allclose(recon._initial_est, random_init)


@pytest.mark.parametrize("algorithm", standard_algos)
def test_recon_numpy(algorithm):
    for gray in [True, False]:
        for dtype in dtypes:
            psf, data = load_data(
                psf_fp=psf_fp,
                data_fp=data_fp,
                downsample=downsample,
                plot=False,
                gray=gray,
                dtype=dtype,
                use_torch=False,
            )
            recon = algorithm(psf, dtype=dtype)
            recon.set_data(data)
            res = recon.apply(n_iter=_n_iter, disp_iter=None, plot=False)
            assert len(psf.shape) == 4
            assert psf.shape[3] == (1 if gray else 3)
            assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


@pytest.mark.parametrize("algorithm", standard_algos + trainable_algos)
def test_recon_torch(algorithm):
    if not torch_is_available:
        return
    for gray in [True, False]:
        for dtype in dtypes:
            psf, data = load_data(
                psf_fp=psf_fp,
                data_fp=data_fp,
                downsample=downsample,
                plot=False,
                gray=gray,
                dtype=dtype,
                use_torch=True,
            )
            recon = algorithm(psf, dtype=dtype, n_iter=_n_iter)
            recon.set_data(data)
            res = recon.apply(disp_iter=None, plot=False)
            assert recon._n_iter == _n_iter
            assert len(psf.shape) == 4
            assert psf.shape[3] == (1 if gray else 3)
            assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"


def test_apgd():
    if pycsou_available:
        for gray in [True, False]:
            for dtype in dtypes:
                for diff_penalty in [APGDPriors.L2, None]:
                    for prox_penalty in [APGDPriors.NONNEG, APGDPriors.L1]:
                        psf, data = load_data(
                            psf_fp=psf_fp,
                            data_fp=data_fp,
                            downsample=downsample,
                            plot=False,
                            gray=gray,
                            dtype=dtype,
                            use_torch=False,
                        )
                        recon = APGD(
                            psf,
                            dtype=dtype,
                            prox_penalty=prox_penalty,
                            diff_penalty=diff_penalty,
                            rel_error=None,
                        )
                        recon.set_data(data)
                        res = recon.apply(n_iter=_n_iter, disp_iter=None, plot=False)
                        assert psf.shape[3] == (1 if gray else 3)
                        assert res.dtype == psf.dtype, f"Got {res.dtype}, expected {dtype}"
    else:
        print("Pycsou not installed. Skipping APGD test.")


#  trainable algorithms
@pytest.mark.parametrize("algorithm", trainable_algos)
def test_trainable_recon(algorithm):
    if torch_is_available:
        for dtype, torch_type in [("float32", torch.float32), ("float64", torch.float64)]:
            psf = torch.rand(1, 32, 64, 3, dtype=torch_type)
            data = torch.rand(2, 1, 32, 64, 3, dtype=torch_type)

            def pre_process(x, **kwargs):
                return x

            def post_process(x, **kwargs):
                return x

            recon = algorithm(
                psf, n_iter=_n_iter, dtype=dtype, pre_process=pre_process, post_process=post_process
            )

            assert (
                next(recon.parameters(), None) is not None
            ), f"{algorithm.__name__} has no trainable parameters"

            res = recon.forward(data)
            loss = torch.mean(res)
            loss.backward()

            assert (
                data.shape[0] == res.shape[0]
            ), f"Batch dimension changed: got {res.shape[0]} expected {data.shape[0]}"

            assert len(psf.shape) == 4
            assert res.shape[4] == 3, "Input in HWC format but output CHW format"


@pytest.mark.parametrize("algorithm", trainable_algos)
def test_trainable_batch(algorithm):
    # test if batch_call and pally give the same result for any batch size
    if not torch_is_available:
        return
    for dtype, torch_type in [("float32", torch.float32), ("float64", torch.float64)]:
        psf = torch.rand(1, 34, 64, 3, dtype=torch_type)
        data1 = torch.rand(5, 1, 34, 64, 3, dtype=torch_type)
        data2 = torch.rand(1, 1, 34, 64, 3, dtype=torch_type)
        data2[0, 0, ...] = data1[0, 0, ...]

        def pre_process(x, **kwargs):
            return x

        def post_process(x, **kwargs):
            return x

        recon = algorithm(
            psf, dtype=dtype, n_iter=_n_iter, pre_process=pre_process, post_process=post_process
        )
        res1 = recon.forward(data1)
        res2 = recon.forward(data2)
        recon.set_data(data2[0])
        res3 = recon.apply(disp_iter=None, plot=False)

        # main test
        torch.testing.assert_close(res1[0, 0, ...], res2[0, 0, ...])
        torch.testing.assert_close(res1[0, 0, ...], res3[0, ...])
        # small test
        assert res1.dtype == psf.dtype, f"Got {res1.dtype}, expected {dtype}"
        assert recon._n_iter == _n_iter
        assert len(psf.shape) == 4
