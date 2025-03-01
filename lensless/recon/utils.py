# #############################################################################
# utils.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import wandb
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import gc
import torch
from torch import nn
from lensless.eval.benchmark import benchmark
from lensless.hardware.trainable_mask import TrainableMask
from tqdm import tqdm
from lensless.recon.drunet.network_unet import UNetRes
from lensless.recon.restormer import Restormer
from lensless.utils.io import save_image
from lensless.utils.plot import plot_image
from lensless.utils.dataset import SimulatedDatasetTrainableMask
from lensless.utils.image import rotate_HWC


def double_cnn_max_pool(c_in, c_out, cnn_kernel=3, max_pool=2, padding=1, skip_last_relu=False):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=cnn_kernel,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=cnn_kernel,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(c_out),
        nn.ReLU() if not skip_last_relu else nn.Identity(),
        # don't pass stride=1, otherwise no pooling/downsampling..
        nn.MaxPool2d(kernel_size=max_pool, padding=0) if max_pool else nn.Identity(),
    )


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, cnn_kernel=3, max_pool=2, padding=1):
        super(ResBlock, self).__init__()
        assert c_in == c_out, "Input and output channels must be the same for residual block."

        # conv layers for residual need to be the same size
        self.double_conv = double_cnn_max_pool(
            c_in, c_in, cnn_kernel=cnn_kernel, max_pool=False, padding=padding, skip_last_relu=True
        )

        # pooling
        self.pooling = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=c_in,
            #     out_channels=c_out,
            #     kernel_size=cnn_kernel,
            #     padding=padding,
            #     bias=False,
            # ),
            # nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool, padding=0),
        )
        # self.pooling = nn.MaxPool2d(kernel_size=max_pool, padding=0)

    def forward(self, x):
        return self.pooling(x + self.double_conv(x))


class CompensationBranch(nn.Module):
    """
    Compensation branch for unrolled algorithm, as in "Robust Reconstruction With Deep Learning to Handle Model Mismatch in Lensless Imaging" (2021).
    """

    def __init__(self, nc, cnn_kernel=3, max_pool=2, in_channel=3, residual=True, padding=1):
        """

        Parameters
        ----------
        nc : list
            Number of channels for each layer of the compensation branch.
        cnn_kernel : int, optional
            Kernel size for convolutional layers, by default 3.
        max_pool : int, optional
            Kernel size for max pooling layers, by default 2.
        in_channel : int, optional
            Number of input channels, by default 3 for RGB.
        residual : bool, optional
            Whether to use residual block or simply double conv for intermediate layers, by default True.
        """
        super(CompensationBranch, self).__init__()

        self.n_iter = len(nc)

        # layers along the compensation branch, f^C in paper
        branch_layers = [
            double_cnn_max_pool(
                in_channel,
                nc[0],
                cnn_kernel=cnn_kernel,
                max_pool=max_pool,
                padding=padding,
            )
        ]
        self.branch_layers = nn.ModuleList(
            branch_layers
            + [
                double_cnn_max_pool(
                    # nc[i] * 2,  # due to concatenation with intermediate layer
                    nc[i] + 3,  # due to concatenation with intermediate layer
                    nc[i + 1],
                    cnn_kernel=cnn_kernel,
                    max_pool=max_pool,
                    padding=padding,
                )
                for i in range(self.n_iter - 1)
            ]
        )

        # residual layers for intermediate output, \tilde{f}^C in paper
        # -- not mentinoed in paper, but added more max-pooling for later residual layers, otherwise dimensions don't match
        self.residual_layers = nn.ModuleList(
            [
                ResBlock(
                    in_channel,
                    in_channel,
                    cnn_kernel=cnn_kernel,
                    max_pool=max_pool ** (i + 1),
                    padding=padding,
                )
                if residual
                else double_cnn_max_pool(
                    in_channel,
                    nc[i],
                    cnn_kernel=cnn_kernel,
                    max_pool=max_pool ** (i + 1),
                    padding=padding,
                )
                for i in range(self.n_iter - 1)
            ]
        )

    def forward(self, x, return_NCHW=True):
        """
        Input must be original input and intermediate outputs: (b, s1, s2, ... , s^{K-1}), where K is the number of iterations.

        See p. 1085 of "Robust Reconstruction With Deep Learning to Handle Model Mismatch in Lensless Imaging" (2021) for more details.
        """
        assert len(x) == self.n_iter, "Input must have the same length as the number of iterations."
        n_depth = x[0].shape[-4]
        h_apo_k = self.branch_layers[0](convert_to_NCHW(x[0]))  # h^{'}_k
        for k in range(self.n_iter - 1):  # eq. 18-21
            # \tilde{h}_k
            h_k = torch.cat([h_apo_k, self.residual_layers[k](convert_to_NCHW(x[k + 1]))], axis=1)
            h_apo_k = self.branch_layers[k + 1](h_k)  # h^{'}_k

        if return_NCHW:
            return h_apo_k
        else:
            return convert_to_NDCHW(h_apo_k, n_depth)


# convert from NDHWC to NCHW
def convert_to_NCHW(image):
    image = image.movedim(-1, -3)
    image = image.reshape(-1, *image.shape[-3:])
    return image


# convert back to NDHWC
def convert_to_NDCHW(image, depth):
    image = image.movedim(-3, -1)
    image = image.reshape(-1, depth, *image.shape[-3:])
    return image


def load_drunet(model_path=None, n_channels=3, requires_grad=False):
    """
    Load a pre-trained Drunet model.

    Parameters
    ----------
    model_path : str, optional
        Path to pre-trained model. Download if not provided.
    n_channels : int
        Number of channels in input image.
    requires_grad : bool
        Whether to require gradients for model parameters.

    Returns
    -------
    model : :py:class:`torch.nn.Module`
        Loaded model.
    """

    if model_path is None:
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(this_file_path, "..", "..", "models", "drunet_color.pth")
        if not os.path.exists(model_path):
            try:
                from torchvision.datasets.utils import download_url
            except ImportError:
                exit()
            msg = "Do you want to download the pretrained DRUNet model (130MB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            output_path = os.path.join(this_file_path, "..", "..", "models")
            if valid:
                url = "https://drive.switch.ch/index.php/s/jTdeMHom025RFRQ/download"
                filename = "drunet_color.pth"
                download_url(url, output_path, filename=filename)

    assert os.path.exists(model_path), f"Model path {model_path} does not exist"

    model = UNetRes(
        in_nc=n_channels + 1,
        out_nc=n_channels,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    )
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = requires_grad

    return model


def apply_denoiser(
    model, image, noise_level=10, mode="inference", compensation_output=None, background=None
):
    """
    Apply a pre-trained denoising model with input in the format Channel, Height, Width.
    An additionnal channel is added for the noise level as done in Drunet.

    Parameters
    ----------
    model : :py:class:`torch.nn.Module`
        Drunet compatible model. Its input must consist of 4 channels (RGB + noise level) and output an RGB image both in CHW format.
    image : :py:class:`torch.Tensor`
        Input image.
    noise_level : float or :py:class:`torch.Tensor`
        Noise level in the image within [0, 255].
    background : :py:class:`torch.Tensor`, optional
        If provided, use background as noise channel instead of noise level.
    device : str
        Device to use for computation. Can be "cpu" or "cuda".
    mode : str
        Mode to use for model. Can be "inference" or "train".

    Returns
    -------
    image : :py:class:`torch.Tensor`
        Reconstructed image.
    """
    assert noise_level > 0
    assert noise_level <= 255

    # convert from NDHWC to NCHW
    depth = image.shape[-4]
    image = image.movedim(-1, -3)
    image = image.reshape(-1, *image.shape[-3:])

    # pad image H and W to next multiple of 8
    top = (8 - image.shape[-2] % 8) // 2
    bottom = (8 - image.shape[-2] % 8) - top
    left = (8 - image.shape[-1] % 8) // 2
    right = (8 - image.shape[-1] % 8) - left
    image = torch.nn.functional.pad(image, (left, right, top, bottom), mode="constant", value=0)

    # use background if provided
    # TODO distinguish between integrated background subtraction where it gets passed to the model instead
    # -- check model.background_subtraction ? At the moment, integrated background subtraction isn't supported
    if background is not None:
        # -- pad
        background = background.movedim(-1, -3)
        background = background.reshape(-1, *background.shape[-3:])
        background = torch.nn.functional.pad(
            background, (left, right, top, bottom), mode="constant", value=0
        )

        # -- concatenate as noise channel
        image = torch.cat((image, background), dim=1)

    else:

        # add noise level as extra channel
        if isinstance(noise_level, torch.Tensor):
            noise_level = noise_level / 255.0
        else:
            noise_level = torch.tensor([noise_level / 255.0])

        image = torch.cat(
            (
                image,
                noise_level.repeat(image.shape[0], 1, image.shape[2], image.shape[3]),
            ),
            dim=1,
        )

    # apply model
    if mode == "inference":
        with torch.no_grad():
            image = model(image, compensation_output)
    elif mode == "train":
        image = model(image, compensation_output)
    else:
        raise ValueError("mode must be 'inference' or 'train'")

    # remove padding
    image = image[:, :, top:-bottom, left:-right]
    # convert back to NDHWC
    image = image.movedim(-3, -1)
    image = image.reshape(-1, depth, *image.shape[-3:])

    return image


def get_drunet_function(model, mode="inference"):
    """
    Return a processing function that applies the DruNet model to an image.
    Legacy function to work with pre-trained models, use get_drunet_function_v2 instead.

    Parameters
    ----------
    model : :py:class:`torch.nn.Module`
        DruNet like denoiser model
    device : str
        Device to use for computation. Can be "cpu" or "cuda".
    mode : str
        Mode to use for model. Can be "inference" or "train".
    """

    def process(image, noise_level=10, compensation_output=None, background=None):
        assert compensation_output is None, "Compensation output not supported for legacy models."
        assert background is None, "Background not supported for legacy models."
        x_max = torch.amax(image, dim=(-2, -3), keepdim=True) + 1e-6
        image = apply_denoiser(
            model,
            image,
            noise_level=noise_level,
            mode=mode,
        )
        image = torch.clip(image, min=0.0) * x_max
        return image

    return process


def get_drunet_function_v2(model, mode="inference"):
    """
    Return a processing function that applies the DruNet model to an image.

    Parameters
    ----------
    model : :py:class:`torch.nn.Module`
        DruNet like denoiser model
    mode : str
        Mode to use for model. Can be "inference" or "train".
    """

    def process(image, noise_level=10, compensation_output=None, background=None):
        x_max = torch.amax(image, dim=(-1, -2, -3, -4), keepdim=True) + 1e-6
        image = apply_denoiser(
            model,
            image / x_max,
            noise_level=noise_level,
            mode=mode,
            compensation_output=compensation_output,
            background=background,
        )
        image = torch.clip(image, min=0.0) * x_max.to(image.device)
        return image

    return process


def measure_gradient(model):
    """
    Helper function to measure L2 norm of the gradient of a model.

    Parameters
    ----------
    model : :py:class:`torch.nn.Module`
        Model to measure gradient of.

    Returns
    -------
    Float
        L2 norm of the gradient of the model.
    """
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def create_process_network(
    network,
    device="cpu",
    device_ids=None,
    concatenate_compensation=False,
    background_subtraction=False,
    input_background=False,
    # unet parameters
    depth=4,
    nc=None,
    # restormer parameters
    restormer_params=None,
):
    """
    Helper function to create a process network.

    Parameters
    ----------
    network : str
        Name of network to use. Can be "DruNet" or "UnetRes".
    depth : int
        Depth of network.
    device : str
        Device to use for computation. Can be "cpu" or "cuda". Defaults to "cpu".
    concatenate_compensation : int
        Number of channels in last layer of compensation branch.

    Returns
    -------
    :py:class:`torch.nn.Module`
        New process network. Already trained for Drunet.
    """

    if nc is None:
        nc = [64, 128, 256, 512]
    else:
        assert len(nc) == 4

    if network == "DruNet":
        assert (
            concatenate_compensation is False
        ), "DruNet does not support concatenation of compensation branch."
        assert background_subtraction is False, "DruNet does not support background subtraction."
        from lensless.recon.utils import load_drunet

        process = load_drunet(requires_grad=True)
        process_name = "DruNet"
    elif network == "UnetRes":
        n_channels = 3
        process = UNetRes(
            in_nc=n_channels * 2
            if input_background
            else n_channels + 1,  # extra channel(s) for noise level(s)
            out_nc=n_channels,
            nc=nc,
            nb=depth,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
            concatenate_compensation=concatenate_compensation,
            input_background=input_background or background_subtraction,
            background_subtraction=background_subtraction,
        )
        process_name = "UnetRes_d" + str(depth)

    elif network == "Restormer":
        assert restormer_params is not None
        process = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=restormer_params["dim"],
            num_blocks=restormer_params["num_blocks"],
            num_refinement_blocks=restormer_params["num_refinement_blocks"],
            heads=restormer_params["heads"],
            ffn_expansion_factor=restormer_params["ffn_expansion_factor"],
            bias=False,
            LayerNorm_type="BiasFree",  # "WithBias"
            dual_pixel_task=False,
        )

        ## -- pretrained
        # process = Restormer(
        #     inp_channels=3,
        #     out_channels=3,
        #     dim=48,
        #     num_blocks=[4, 6, 6, 8],
        #     num_refinement_blocks=4,
        #     heads=[1, 2, 4, 8],
        #     ffn_expansion_factor=2.66,
        #     bias=False,
        #     LayerNorm_type="BiasFree",  # "WithBias"
        #     dual_pixel_task=False,
        # )
        # # TODO download path from https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0
        # weights_path = "/root/LenslessPiCam/notebook/real_denoising.pth"
        # checkpoint = torch.load(weights_path)
        # process.load_state_dict(checkpoint['params'])

        process_name = "Restormer"

    else:
        process = None
        process_name = None

    if process is not None:
        if device_ids is not None:
            process = torch.nn.DataParallel(process, device_ids=device_ids)
        process = process.to(device)

    return (process, process_name)


class Trainer:
    def __init__(
        self,
        recon,
        train_dataset,
        test_dataset,
        test_size=0.15,
        mask=None,
        batch_size=4,
        eval_batch_size=10,
        loss="l2",
        lpips=None,
        l1_mask=None,
        optimizer=None,
        skip_NAN=False,
        algorithm_name="Unknown",
        metric_for_best_model=None,
        save_every=None,
        gamma=None,
        logger=None,
        crop=None,
        clip_grad=1.0,
        unrolled_output_factor=False,
        random_rotate=False,
        random_shift=False,
        pre_proc_aux=False,
        extra_eval_sets=None,
        use_wandb=False,
        # for adding components during training
        pre_process=None,
        pre_process_delay=None,
        pre_process_freeze=None,
        pre_process_unfreeze=None,
        post_process=None,
        post_process_delay=None,
        post_process_freeze=None,
        post_process_unfreeze=None,
        n_epoch=None,
    ):
        """
        Class to train a reconstruction algorithm. Inspired by Trainer from `HuggingFace <https://huggingface.co/docs/transformers/main_classes/trainer>`__.

        The train and test metrics at the end of each epoch can be found in ``self.metrics``,
        with "LOSS" being the train loss. The test loss can be found in "MSE" (if loss is "l2") or
        "MAE" (if loss is "l1"). If ``lpips`` is not None, the LPIPS loss is also added
        to the train loss, such that the test loss can be computed as "MSE" + ``lpips`` * "LPIPS_Vgg"
        (or "MAE" + ``lpips`` * "LPIPS_Vgg").

        Parameters
        ----------
        recon : :py:class:`lensless.TrainableReconstructionAlgorithm`
            Reconstruction algorithm to train.
        train_dataset : :py:class:`torch.utils.data.Dataset`
            Dataset to use for training.
        test_dataset : :py:class:`torch.utils.data.Dataset`
            Dataset to use for testing.
        test_size : float, optional
            If test_dataset is None, fraction of the train dataset to use for testing, by default 0.15.
        mask : TrainableMask, optional
            Trainable mask to use for training. If none, training with fix psf, by default None.
        batch_size : int, optional
            Batch size to use for training, by default 4.
        eval_batch_size : int, optional
            Batch size to use for evaluation, by default 10.
        loss : str, optional
            Loss function to use for training "l1" or "l2", by default "l2".
        lpips : float, optional
            the weight of the lpips(VGG) in the total loss. If None ignore. By default None.
        l1_mask : float, optional
            the weight of the l1 norm of the mask in the total loss. If None ignore. By default None.
        optimizer : dict
            Optimizer configuration.
        skip_NAN : bool, optional
            Whether to skip update if any gradiant are NAN (True) or to throw an error(False), by default False
        algorithm_name : str, optional
            Algorithm name for logging, by default "Unknown".
        metric_for_best_model : str, optional
            Metric to use for saving the best model. If None, will default to evaluation loss. Default is None.
        save_every : int, optional
            Save model every ``save_every`` epochs. If None, just save best model.
        gamma : float, optional
            Gamma correction to apply to PSFs when plotting. If None, no gamma correction is applied. Default is None.
        logger : :py:class:`logging.Logger`, optional
            Logger to use for logging. If None, just print to terminal. Default is None.
        crop : dict, optional
            Crop to apply to images before computing loss (by applying a mask). If None, no crop is applied. Default is None.
        unrolled_output_factor : float, optional
            How much of the unrolled loss to add to the total loss. If False, no unrolled loss is added. Default is False. Only applicable if a post-processor is used.
        pre_process : :py:class:`torch.nn.Module`, optional
            Pre process component to add during training. Default is None.
        pre_process_delay : int, optional
            Epoch at which to add pre process component. Default is None.
        pre_process_freeze : int, optional
            Epoch at which to freeze pre process component. Default is None.
        pre_process_unfreeze : int, optional
            Epoch at which to unfreeze pre process component. Default is None.
        post_process : :py:class:`torch.nn.Module`, optional
            Post process component to add during training. Default is None.
        post_process_delay : int, optional
            Epoch at which to add post process component. Default is None.
        post_process_freeze : int, optional
            Epoch at which to freeze post process component. Default is None.
        post_process_unfreeze : int, optional
            Epoch at which to unfreeze post process component. Default is None.

        """
        global print

        self.use_wandb = use_wandb

        self.device = recon._psf.device
        self.logger = logger
        if self.logger is not None:
            self.print = self.logger.info
        else:
            self.print = print
        self.recon = recon

        self.pre_process = pre_process
        self.pre_process_delay = pre_process_delay
        self.pre_process_freeze = pre_process_freeze
        self.pre_process_unfreeze = pre_process_unfreeze
        self.pre_process_delay = pre_process_delay
        if pre_process_delay is not None:
            assert pre_process is not None

        self.post_process = post_process
        self.post_process_delay = post_process_delay
        self.post_process_freeze = post_process_freeze
        self.post_process_unfreeze = post_process_unfreeze
        self.post_process_delay = post_process_delay
        if post_process_delay is not None:
            assert post_process is not None

        assert train_dataset is not None
        if test_dataset is None:
            assert test_size < 1.0 and test_size > 0.0
            # split train dataset
            train_size = int((1 - test_size) * len(train_dataset))
            test_size = len(train_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, test_size]
            )
            self.print(f"Train size : {train_size}, Test size : {test_size}")

        self.train_dataset = train_dataset
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(self.device != "cpu"),
        )
        self.test_dataset = test_dataset
        self.extra_eval_sets = extra_eval_sets  # additional datasets to evaluate on
        self.lpips = lpips
        self.skip_NAN = skip_NAN
        self.eval_batch_size = eval_batch_size
        self.train_multimask = False
        if hasattr(train_dataset, "multimask"):
            self.train_multimask = train_dataset.multimask
        self.train_random_flip = train_dataset.random_flip
        self.random_rotate = random_rotate
        self.random_shift = random_shift
        if hasattr(train_dataset, "measured_bg"):
            self.background = train_dataset.measured_bg
        else:
            self.background = False
        if self.random_shift:
            raise NotImplementedError("Random shift not implemented yet.")

        # check if Subset and if simulating dataset
        self.simulated_dataset_trainable_mask = False
        if isinstance(self.test_dataset, SimulatedDatasetTrainableMask):
            # assuming the case for both training and testing
            self.simulated_dataset_trainable_mask = True

        self.mask = mask
        self.gamma = gamma
        if mask is not None:
            assert isinstance(mask, TrainableMask)
            self.use_mask = True
        else:
            self.use_mask = False
        if self.use_mask:
            # save original PSF
            psf_np = self.mask.get_psf().detach().cpu().numpy()[0, ...]
            psf_np = psf_np.squeeze()  # remove (potential) singleton color channel
            np.save("psf_original.npy", psf_np)
            fp = "psf_original.png"
            save_image(psf_np, fp)
            plot_image(psf_np, gamma=self.gamma)
            fp_plot = "psf_original_plot.png"
            plt.savefig(fp_plot)

            if self.use_wandb:
                wandb.log({"psf": wandb.Image(fp)}, step=0)
                wandb.log({"psf_plot": wandb.Image(fp_plot)}, step=0)

        self.l1_mask = l1_mask

        # loss
        if loss == "l2":
            self.Loss = torch.nn.MSELoss()
        elif loss == "l1":
            self.Loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unsuported loss : {loss}")

        # -- Lpips loss
        if lpips:
            try:
                import lpips

                self.Loss_lpips = lpips.LPIPS(net="vgg").to(self.device)
            except ImportError:
                return ImportError(
                    "lpips package is need for LPIPS loss. Install using : pip install lpips"
                )

        self.crop = crop

        # -- adding unrolled loss
        self.unrolled_output_factor = unrolled_output_factor
        if self.unrolled_output_factor:
            assert self.unrolled_output_factor > 0
            assert self.post_process is not None
            assert self.post_process_delay is None
            assert self.post_process_unfreeze is None
            assert self.post_process_freeze is None

        # -- adding pre-processed output to loss
        self.pre_proc_aux = pre_proc_aux
        if self.pre_proc_aux:
            assert self.pre_process is not None
            assert self.pre_process_delay is None
            assert self.pre_process_unfreeze is None
            assert self.pre_process_freeze is None

        # optimizer
        self.clip_grad_norm = clip_grad
        self.optimizer_config = optimizer
        self.n_epoch = n_epoch
        self.lr_step_epoch = optimizer.lr_step_epoch
        self.set_optimizer()

        # metrics
        self.metrics = {
            "LOSS": [],  # train loss
            "LOSS_TEST": [],  # test loss
            "MSE": [],
            "MAE": [],
            "LPIPS_Vgg": [],
            "LPIPS_Alex": [],
            "PSNR": [],
            "SSIM": [],
            "ReconstructionError": [],
            "n_iter": self.recon._n_iter,
            "algorithm": algorithm_name,
            "metric_for_best_model": metric_for_best_model,
            "best_epoch": 0,
            "best_eval_score": 0
            if metric_for_best_model == "PSNR" or metric_for_best_model == "SSIM"
            else np.inf,
        }
        if self.unrolled_output_factor:
            # -- add unrolled metrics
            for key in ["MSE", "MAE", "LPIPS_Vgg", "LPIPS_Alex", "PSNR", "SSIM"]:
                self.metrics[key + "_unrolled"] = []
        if self.pre_proc_aux:
            self.metrics[
                "ReconstructionError_PreProc"
            ] = []  # reconstruction error of ||pre_proc(y) - A * camera_inversion(y)||
        if metric_for_best_model is not None:
            assert metric_for_best_model in self.metrics.keys()
        if extra_eval_sets is not None:
            for key in extra_eval_sets:
                self.metrics[key] = dict()
        self.save_every = save_every

        # Backward hook that detect NAN in the gradient and print the layer weights
        if not self.skip_NAN:

            def detect_nan(grad):
                if torch.isnan(grad).any():
                    if self.logger:
                        self.logger.info(grad)
                    else:
                        print(grad, flush=True)
                    for name, param in recon.named_parameters():
                        if param.requires_grad:
                            self.print(name, param)
                    raise ValueError("Gradient is NaN")
                return grad

            for param in recon.parameters():
                if param.requires_grad:
                    param.register_hook(detect_nan)
                    if param.requires_grad:
                        param.register_hook(detect_nan)

    def set_optimizer(self, last_epoch=-1):

        if self.optimizer_config.type == "AdamW":
            print("USING ADAMW")
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": [p for p in self.recon.parameters() if p.dim() > 1]},
                    {
                        "params": [p for p in self.recon.parameters() if p.dim() <= 1],
                        "weight_decay": 0,
                    },  # no weight decay on bias terms
                ],
                lr=self.optimizer_config.lr,
                weight_decay=0.01,
            )
        else:
            print(f"USING {self.optimizer_config.type}")
            parameters = [{"params": self.recon.parameters()}]
            self.optimizer = getattr(torch.optim, self.optimizer_config.type)(
                parameters, lr=self.optimizer_config.lr
            )

        # Scheduler
        if self.optimizer_config.slow_start:

            def learning_rate_function(epoch):
                if epoch == 0:
                    return self.optimizer_config.slow_start
                elif epoch == 1:
                    return math.sqrt(self.optimizer_config.slow_start)
                else:
                    return 1

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=learning_rate_function, last_epoch=last_epoch
            )

        elif self.optimizer_config.final_lr:

            assert self.optimizer_config.final_lr < self.optimizer_config.lr
            assert self.n_epoch is not None

            # # linear decay
            # def learning_rate_function(epoch):
            #     slope = (start / final - 1) / (n_epoch)
            #     return 1 / (1 + slope * epoch)

            # exponential decay
            def learning_rate_function(epoch):
                final_decay = self.optimizer_config.final_lr / self.optimizer_config.lr
                final_decay = final_decay ** (1 / (self.n_epoch - 1))
                return final_decay**epoch

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=learning_rate_function, last_epoch=last_epoch
            )

        elif self.optimizer_config.exp_decay:

            def learning_rate_function(epoch):
                return self.optimizer_config.exp_decay**epoch

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=learning_rate_function, last_epoch=last_epoch
            )

        elif self.optimizer_config.cosine_decay_warmup:

            if self.lr_step_epoch:
                total_iterations = self.n_epoch
            else:
                total_iterations = len(self.train_dataloader) * self.n_epoch
            warmup_steps = int(0.05 * total_iterations)

            def cosine_decay_with_warmup(step, warmup_steps, total_steps):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: cosine_decay_with_warmup(
                    step, warmup_steps, total_iterations
                ),
            )

        elif self.optimizer_config.step:

            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.optimizer_config.step,
                gamma=self.optimizer_config.gamma,
                last_epoch=last_epoch,
                verbose=True,
            )

        else:

            def learning_rate_function(epoch):
                return 1

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=learning_rate_function, last_epoch=last_epoch
            )

    def train_epoch(self, data_loader):
        """
        Train for one epoch.

        Parameters
        ----------
        data_loader : :py:class:`torch.utils.data.DataLoader`
            Data loader to use for training.

        Returns
        -------
        float
            Mean loss of the epoch.
        """
        mean_loss = 0.0
        i = 1.0
        pbar = tqdm(data_loader)
        self.recon.train()
        for batch in pbar:

            # get batch
            flip_lr = None
            flip_ud = None
            background = None
            X = batch[0].to(self.device)
            y = batch[1].to(self.device)
            if self.background:
                background = batch[-1].to(self.device)
            if self.train_random_flip or self.train_multimask:
                psfs = batch[2].to(self.device)
            else:
                psfs = None
            if self.train_random_flip:
                flip_lr = batch[3]
                flip_ud = batch[4]

            random_rotate = False
            if self.random_rotate:
                random_rotate = np.random.uniform(-self.random_rotate, self.random_rotate)
                X = rotate_HWC(X, random_rotate)
                y = rotate_HWC(y, random_rotate)
                if psfs is None:
                    psf_single = self.recon._psf
                    psf_single = rotate_HWC(psf_single, random_rotate)
                    self.recon._set_psf(psf_single.to(self.device))
                else:
                    psfs = rotate_HWC(psfs, random_rotate)

            # update psf according to mask
            if self.use_mask:
                self.recon._set_psf(self.mask.get_psf().to(self.device))

            # forward pass
            # torch.autograd.set_detect_anomaly(True)    # for debugging
            y_pred = self.recon.forward(batch=X, psfs=psfs, background=background)
            if self.unrolled_output_factor or self.pre_proc_aux:
                y_pred, camera_inv_out, pre_proc_out = y_pred[0], y_pred[1], y_pred[2]

            # normalizing each output
            eps = 1e-12
            y_pred_max = torch.amax(y_pred, dim=(-1, -2, -3), keepdim=True) + eps
            y_pred = y_pred / y_pred_max

            # normalizing y
            y_max = torch.amax(y, dim=(-1, -2, -3), keepdim=True) + eps
            y = y / y_max

            # convert to CHW for loss and remove depth
            y_pred_crop = y_pred.reshape(-1, *y_pred.shape[-3:]).movedim(-1, -3)
            y = y.reshape(-1, *y.shape[-3:]).movedim(-1, -3)

            # extraction region of interest for loss
            if hasattr(self.train_dataset, "alignment"):
                if self.train_dataset.alignment is not None:
                    y_pred_crop = self.train_dataset.extract_roi(
                        y_pred_crop,
                        axis=(-2, -1),
                        flip_lr=flip_lr,
                        flip_ud=flip_ud,
                        rotate_aug=random_rotate,
                    )
                else:
                    y_pred_crop, y = self.train_dataset.extract_roi(
                        y_pred_crop,
                        axis=(-2, -1),
                        lensed=y,
                        flip_lr=flip_lr,
                        flip_ud=flip_ud,
                        rotate_aug=random_rotate,
                    )

            elif self.crop is not None:
                assert flip_lr is None and flip_ud is None
                y_pred_crop = y_pred_crop[
                    ...,
                    self.crop["vertical"][0] : self.crop["vertical"][1],
                    self.crop["horizontal"][0] : self.crop["horizontal"][1],
                ]
                y = y[
                    ...,
                    self.crop["vertical"][0] : self.crop["vertical"][1],
                    self.crop["horizontal"][0] : self.crop["horizontal"][1],
                ]

            loss_v = self.Loss(y_pred_crop, y)

            # add LPIPS loss
            if self.lpips:

                if y_pred_crop.shape[1] == 1:
                    # if only one channel, repeat for LPIPS
                    y_pred_crop = y_pred_crop.repeat(1, 3, 1, 1)
                    y = y.repeat(1, 3, 1, 1)

                # value for LPIPS needs to be in range [-1, 1]
                loss_v = loss_v + self.lpips * torch.mean(
                    self.Loss_lpips(2 * y_pred_crop - 1, 2 * y - 1)
                )
            if self.use_mask and self.l1_mask:
                for p in self.mask.parameters():
                    if p.requires_grad:
                        loss_v = loss_v + self.l1_mask * torch.mean(torch.abs(p))

            if self.unrolled_output_factor:
                # -- normalize
                unrolled_out_max = torch.amax(camera_inv_out, dim=(-1, -2, -3), keepdim=True) + eps
                camera_inv_out_norm = camera_inv_out / unrolled_out_max

                # -- convert to CHW for loss and remove depth
                camera_inv_out_norm = camera_inv_out_norm.reshape(
                    -1, *camera_inv_out.shape[-3:]
                ).movedim(-1, -3)

                # -- extraction region of interest for loss
                if hasattr(self.train_dataset, "alignment"):
                    if self.train_dataset.alignment is not None:
                        camera_inv_out_norm = self.train_dataset.extract_roi(
                            camera_inv_out_norm, axis=(-2, -1)
                        )
                    else:
                        camera_inv_out_norm = self.train_dataset.extract_roi(
                            camera_inv_out_norm,
                            axis=(-2, -1),
                            # y=y   # lensed already extracted before
                        )
                    assert np.all(y.shape == camera_inv_out_norm.shape)
                elif self.crop is not None:
                    camera_inv_out_norm = camera_inv_out_norm[
                        ...,
                        self.crop["vertical"][0] : self.crop["vertical"][1],
                        self.crop["horizontal"][0] : self.crop["horizontal"][1],
                    ]

                # -- compute unrolled output loss
                loss_unrolled = self.Loss(camera_inv_out_norm, y)

                # -- add LPIPS loss
                if self.lpips:
                    if camera_inv_out_norm.shape[1] == 1:
                        # if only one channel, repeat for LPIPS
                        camera_inv_out_norm = camera_inv_out_norm.repeat(1, 3, 1, 1)

                    # value for LPIPS needs to be in range [-1, 1]
                    loss_unrolled = loss_unrolled + self.lpips * torch.mean(
                        self.Loss_lpips(2 * camera_inv_out_norm - 1, 2 * y - 1)
                    )

                # -- add unrolled loss to total loss
                loss_v = loss_v + self.unrolled_output_factor * loss_unrolled

            if self.pre_proc_aux:
                # -- normalize
                unrolled_out_max = torch.amax(camera_inv_out, dim=(-1, -2, -3), keepdim=True) + eps
                camera_inv_out_norm = camera_inv_out / unrolled_out_max

                err = torch.mean(
                    self.recon.reconstruction_error(
                        prediction=camera_inv_out_norm,
                        # prediction=y_pred,
                        lensless=pre_proc_out,
                    )
                )
                loss_v = loss_v + self.pre_proc_aux * err

            # backward pass
            loss_v.backward()

            # check mask parameters are learning
            if self.use_mask:
                for p in self.mask.parameters():
                    assert p.grad is not None

            if self.clip_grad_norm is not None:
                if self.use_mask:
                    torch.nn.utils.clip_grad_norm_(self.mask.parameters(), self.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.recon.parameters(), self.clip_grad_norm)

            # if any gradient is NaN, skip training step
            if self.skip_NAN:
                recon_is_NAN = False
                mask_is_NAN = False
                for param in self.recon.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        recon_is_NAN = True
                        break
                if self.use_mask:
                    for param in self.mask.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            mask_is_NAN = True
                            break
                if recon_is_NAN or mask_is_NAN:
                    if recon_is_NAN:
                        self.print(
                            "NAN detected in reconstruction gradient, skipping training step"
                        )
                    if mask_is_NAN:
                        self.print("NAN detected in mask gradient, skipping training step")
                    i += 1
                    continue

            self.optimizer.step()
            if not self.lr_step_epoch:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            # update mask
            if self.use_mask:
                self.mask.update_mask()
                if self.simulated_dataset_trainable_mask:
                    self.train_dataloader.dataset.set_psf()

            mean_loss += (loss_v.item() - mean_loss) * (1 / i)
            pbar.set_description(f"loss : {mean_loss}")
            i += 1

        self.print(f"loss : {mean_loss}")

        return mean_loss

    def evaluate(self, mean_loss, epoch, disp=None):
        """
        Evaluate the reconstruction algorithm on the test dataset.

        Parameters
        ----------
        mean_loss : float
            Mean loss of the last epoch.
        disp : list of int, optional
            Test set examples to visualize at the end of each epoch, by default None.
        """
        if self.test_dataset is None:
            return

        if self.use_mask and self.simulated_dataset_trainable_mask:
            with torch.no_grad():
                self.test_dataset.set_psf()

        output_dir = None
        if disp is not None:
            output_dir = os.path.join("eval_recon")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_dir = os.path.join(output_dir, str(epoch))

        # benchmarking
        self.recon.eval()
        current_metrics = benchmark(
            self.recon,
            self.test_dataset,
            batchsize=self.eval_batch_size,
            save_idx=disp,
            output_dir=output_dir,
            crop=self.crop,
            unrolled_output_factor=self.unrolled_output_factor,
            pre_process_aux=self.pre_proc_aux,
            use_wandb=self.use_wandb,
            epoch=epoch,
        )

        # update metrics with current metrics
        self.metrics["LOSS"].append(mean_loss)
        if self.use_wandb:
            wandb.log({"LOSS": mean_loss}, step=epoch)
        for key in current_metrics:
            self.metrics[key].append(current_metrics[key])

        # check best metric
        if self.metrics["metric_for_best_model"] is None:
            eval_loss = current_metrics["MSE"]
            if self.lpips is not None:
                eval_loss += self.lpips * current_metrics["LPIPS_Vgg"]
            if self.use_mask and self.l1_mask:
                with torch.no_grad():
                    for p in self.mask.parameters():
                        if p.requires_grad:
                            eval_loss += self.l1_mask * np.mean(np.abs(p.cpu().detach().numpy()))
            if self.unrolled_output_factor:
                unrolled_loss = current_metrics["MSE_unrolled"]
                if self.lpips is not None:
                    unrolled_loss += self.lpips * current_metrics["LPIPS_Vgg_unrolled"]
                eval_loss += self.unrolled_output_factor * unrolled_loss
            if self.pre_proc_aux:
                eval_loss += self.pre_proc_aux * current_metrics["ReconstructionError_PreProc"]
        else:
            eval_loss = current_metrics[self.metrics["metric_for_best_model"]]

        self.metrics["LOSS_TEST"].append(eval_loss)
        if self.use_wandb:
            wandb.log({"LOSS_TEST": eval_loss}, step=epoch)

        # add extra evaluation sets
        extra_metrics_epoch = {}
        if self.extra_eval_sets is not None:
            for eval_set in self.extra_eval_sets:

                # create output directory
                output_dir = None
                if disp is not None:
                    output_dir = os.path.join("eval_recon")
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    output_dir = os.path.join(output_dir, str(epoch) + f"_{eval_set}")

                if hasattr(self.extra_eval_sets[eval_set], "multimask"):
                    if not self.extra_eval_sets[eval_set].multimask:
                        # need to set correct PSF for evaluation
                        # TODO cleaner way to set PSF?
                        self.recon._set_psf(self.extra_eval_sets[eval_set].psf.to(self.device))

                # benchmarking
                extra_metrics = benchmark(
                    self.recon,
                    self.extra_eval_sets[eval_set],
                    batchsize=self.eval_batch_size,
                    save_idx=disp,
                    output_dir=output_dir,
                    crop=self.crop,
                    unrolled_output_factor=self.unrolled_output_factor,
                    use_wandb=self.use_wandb,
                    label=eval_set,
                    epoch=epoch,
                )

                # add metrics to dictionary
                for key in extra_metrics:
                    if key not in self.metrics[eval_set]:
                        self.metrics[eval_set][key] = [extra_metrics[key]]
                    else:
                        self.metrics[eval_set][key].append(extra_metrics[key])
                    extra_metrics_epoch[f"{eval_set}_{key}"] = extra_metrics[key]

            # set back PSF to original in case changed
            # TODO: cleaner way?
            if not self.train_multimask:
                self.recon._set_psf(self.train_dataset.psf.to(self.device))

        # log metrics to wandb
        if self.use_wandb:
            wandb.log(current_metrics, step=epoch)
            if self.extra_eval_sets is not None:
                wandb.log(extra_metrics_epoch, step=epoch)

        return eval_loss

    def on_epoch_end(self, mean_loss, save_pt, epoch, disp=None):
        """
        Called at the end of each epoch.

        Parameters
        ----------
        mean_loss : float
            Mean loss of the last epoch.
        save_pt : str
            Path to save metrics dictionary to. If None, no logging of metrics.
        epoch : int
            Current epoch.
        disp : list of int, optional
            Test set examples to visualize at the end of each epoch, by default None.
        """
        if save_pt is None:
            # Use current directory
            save_pt = os.getcwd()

        # save model
        epoch_eval_metric = self.evaluate(mean_loss, epoch, disp=disp)
        new_best = False
        if (
            self.metrics["metric_for_best_model"] == "PSNR"
            or self.metrics["metric_for_best_model"] == "SSIM"
        ):
            if epoch_eval_metric > self.metrics["best_eval_score"]:
                self.metrics["best_eval_score"] = epoch_eval_metric
                new_best = True
        else:
            if epoch_eval_metric < self.metrics["best_eval_score"]:
                self.metrics["best_eval_score"] = epoch_eval_metric
                new_best = True

        if new_best:
            self.metrics["best_epoch"] = epoch
            self.save(path=save_pt, include_optimizer=False, epoch="BEST")

        if self.save_every is not None and epoch % self.save_every == 0:
            self.save(path=save_pt, include_optimizer=False, epoch=epoch)

        # save dictionary metrics to file with json
        with open(os.path.join(save_pt, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def train(self, n_epoch=1, save_pt=None, disp=None):
        """
        Train the reconstruction algorithm.

        Parameters
        ----------
        n_epoch : int, optional
            Number of epochs to train for, by default 1
        save_pt : str, optional
            Path to save metrics dictionary to. If None, use current directory, by default None
        disp : list of int, optional
            test set examples to visualize at the end of each epoch, by default None.
        """

        start_time = time.time()

        self.evaluate(mean_loss=1, epoch=0, disp=disp)
        for epoch in range(n_epoch):

            # add extra components (if specified)
            changing_n_param = False
            if epoch == self.pre_process_delay:
                self.print("Adding pre process component")
                self.recon.set_pre_process(self.pre_process)
                changing_n_param = True
            if epoch == self.post_process_delay:
                self.print("Adding post process component")
                self.recon.set_post_process(self.post_process)
                changing_n_param = True
            if epoch == self.pre_process_freeze:
                self.print("Freezing pre process")
                self.recon.freeze_pre_process()
                changing_n_param = True
            if epoch == self.post_process_freeze:
                self.print("Freezing post process")
                self.recon.freeze_post_process()
                changing_n_param = True
            if epoch == self.pre_process_unfreeze:
                self.print("Unfreezing pre process")
                self.recon.unfreeze_pre_process()
                changing_n_param = True
            if epoch == self.post_process_unfreeze:
                self.print("Unfreezing post process")
                self.recon.unfreeze_post_process()
                changing_n_param = True

            # count number of parameters with requires_grad = True
            if changing_n_param:
                n_param = sum(p.numel() for p in self.recon.parameters() if p.requires_grad)
                if self.mask is not None:
                    n_param += sum(p.numel() for p in self.mask.parameters() if p.requires_grad)
                self.print(f"Training {n_param} parameters")

            self.print(f"Epoch {epoch} with learning rate {self.scheduler.get_last_lr()}")
            mean_loss = self.train_epoch(self.train_dataloader)
            # offset because of evaluate before loop
            self.on_epoch_end(mean_loss, save_pt, epoch + 1, disp=disp)
            if self.lr_step_epoch:
                self.scheduler.step()

            torch.cuda.empty_cache()
            gc.collect()


        self.print(f"Train time [hour] : {(time.time() - start_time) / 3600} h")

    def save(self, epoch, path="recon", include_optimizer=False):
        # create directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # save mask parameters
        if self.use_mask:

            for name, param in self.mask.named_parameters():
                # save as numpy array
                if param.requires_grad:
                    np.save(
                        os.path.join(path, f"mask{name}_epoch{epoch}.npy"),
                        param.cpu().detach().numpy(),
                    )

            torch.save(
                self.mask._optimizer.state_dict(), os.path.join(path, f"mask_optim_epoch{epoch}.pt")
            )

            psf_np = self.mask.get_psf().detach().cpu().numpy()[0, ...]
            psf_np = psf_np.squeeze()  # remove (potential) singleton color channel
            np.save(os.path.join(path, f"psf_epoch{epoch}.npy"), psf_np)
            fp = os.path.join(path, f"psf_epoch{epoch}.png")
            save_image(psf_np, fp)
            plot_image(psf_np, gamma=self.gamma)
            fp_plot = os.path.join(path, f"psf_epoch{epoch}_plot.png")
            plt.savefig(fp_plot)

            if self.use_wandb and epoch != "BEST":
                wandb.log({"psf": wandb.Image(fp)}, step=epoch)
                wandb.log({"psf_plot": wandb.Image(fp_plot)}, step=epoch)

            if epoch == "BEST":
                # save difference with original PSF
                psf_original = np.load("psf_original.npy")
                diff = psf_np - psf_original
                np.save(os.path.join(path, "psf_epochBEST_diff.npy"), diff)
                diff_abs = np.abs(diff)
                save_image(diff_abs, os.path.join(path, "psf_epochBEST_diffabs.png"))
                _, ax = plt.subplots()
                im = ax.imshow(diff_abs, cmap="gray" if diff_abs.ndim == 2 else None)
                plt.colorbar(im, ax=ax)
                ax.set_title("Absolute difference with original PSF")
                plt.savefig(os.path.join(path, "psf_epochBEST_diffabs_plot.png"))

        # save optimizer
        if include_optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(path, f"optim_epoch{epoch}.pt"))

        # save recon
        torch.save(self.recon.state_dict(), os.path.join(path, f"recon_epoch{epoch}"))
