# #############################################################################
# utils.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################


import json
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from hydra.utils import get_original_cwd
import os
import torch
from lensless.eval.benchmark import benchmark
from lensless.hardware.trainable_mask import TrainableMask
from tqdm import tqdm
from lensless.recon.drunet.network_unet import UNetRes
from lensless.utils.io import save_image
from lensless.utils.plot import plot_image


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
        model_path = os.path.join(get_original_cwd(), "models", "drunet_color.pth")
        if not os.path.exists(model_path):
            try:
                from torchvision.datasets.utils import download_url
            except ImportError:
                exit()
            msg = "Do you want to download the pretrained DRUNet model (130MB)?"

            # default to yes if no input is given
            valid = input("%s (Y/n) " % msg).lower() != "n"
            output_path = os.path.join(get_original_cwd(), "models")
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


def apply_denoiser(model, image, noise_level=10, device="cpu", mode="inference"):
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
        Noise level in the image.
    device : str
        Device to use for computation. Can be "cpu" or "cuda".
    mode : str
        Mode to use for model. Can be "inference" or "train".

    Returns
    -------
    image : :py:class:`torch.Tensor`
        Reconstructed image.
    """
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
    # add noise level as extra channel
    image = image.to(device)
    if isinstance(noise_level, torch.Tensor):
        noise_level = noise_level / 255.0
    else:
        noise_level = torch.tensor([noise_level / 255.0]).to(device)
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
            image = model(image)
    elif mode == "train":
        image = model(image)
    else:
        raise ValueError("mode must be 'inference' or 'train'")

    # remove padding
    image = image[:, :, top:-bottom, left:-right]
    # convert back to NDHWC
    image = image.movedim(-3, -1)
    image = image.reshape(-1, depth, *image.shape[-3:])
    return image


def get_drunet_function(model, device="cpu", mode="inference"):
    """
    Return a porcessing function that applies the DruNet model to an image.

    Parameters
    ----------
    model : :py:class:`torch.nn.Module`
        DruNet like denoiser model
    device : str
        Device to use for computation. Can be "cpu" or "cuda".
    mode : str
        Mode to use for model. Can be "inference" or "train".
    """

    def process(image, noise_level):
        x_max = torch.amax(image, dim=(-2, -3), keepdim=True) + 1e-6
        image = apply_denoiser(
            model,
            image,
            noise_level=noise_level,
            device=device,
            mode=mode,
        )
        image = torch.clip(image, min=0.0) * x_max
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


def create_process_network(network, depth, device="cpu", nc=None):
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
        from lensless.recon.utils import load_drunet

        process = load_drunet(requires_grad=True).to(device)
        process_name = "DruNet"
    elif network == "UnetRes":
        from lensless.recon.drunet.network_unet import UNetRes

        n_channels = 3
        process = UNetRes(
            in_nc=n_channels + 1,
            out_nc=n_channels,
            nc=nc,
            nb=depth,
            act_mode="R",
            downsample_mode="strideconv",
            upsample_mode="convtranspose",
        ).to(device)
        process_name = "UnetRes_d" + str(depth)
    else:
        process = None
        process_name = None

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
        # for adding components during training
        pre_process=None,
        pre_process_delay=None,
        pre_process_freeze=None,
        pre_process_unfreeze=None,
        post_process=None,
        post_process_delay=None,
        post_process_freeze=None,
        post_process_unfreeze=None,
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
        if pre_process_delay is not None:
            assert pre_process is not None
        else:
            self.pre_process_delay = -1

        self.post_process = post_process
        self.post_process_delay = post_process_delay
        self.post_process_freeze = post_process_freeze
        self.post_process_unfreeze = post_process_unfreeze
        if post_process_delay is not None:
            assert post_process is not None
        else:
            self.post_process_delay = -1

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

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(self.device != "cpu"),
        )
        self.test_dataset = test_dataset
        self.lpips = lpips
        self.skip_NAN = skip_NAN
        self.eval_batch_size = eval_batch_size

        self.mask = mask
        if mask is not None:
            assert isinstance(mask, TrainableMask)
            self.use_mask = True
        else:
            self.use_mask = False

        self.l1_mask = l1_mask
        self.gamma = gamma

        # loss
        if loss == "l2":
            self.Loss = torch.nn.MSELoss()
        elif loss == "l1":
            self.Loss = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unsuported loss : {loss}")

        # Lpips loss
        if lpips:
            try:
                import lpips

                self.Loss_lpips = lpips.LPIPS(net="vgg").to(self.device)
                
            except ImportError:
                return ImportError(
                    "lpips package is need for LPIPS loss. Install using : pip install lpips"
                )

        self.crop = crop

        # optimizer
        self.clip_grad_norm = clip_grad
        self.optimizer_config = optimizer
        self.set_optimizer()
        
        self.metrics = {
            "LOSS": [],  # train loss
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
        if metric_for_best_model is not None:
            assert metric_for_best_model in self.metrics.keys()
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

        # if self.optimizer_config.type == "Adam":
        #     parameters = [{"params": self.recon.parameters()}]
        #     self.optimizer = torch.optim.Adam(parameters, lr=self.optimizer_config.lr)
        # else:
        #     raise ValueError(f"Unsupported optimizer : {self.optimizer_config.type}")
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
        for X, y in pbar:
            # send to device
            X = X.to(self.device)
            y = y.to(self.device)

            # BEFORE
            # # update psf according to mask
            # if self.use_mask:
            #     new_psf = self.mask.get_psf().to(self.device)
            #     self.recon._set_psf(new_psf)

            # forward pass
            y_pred = self.recon.batch_call(X.to(self.device))
            # normalizing each output
            eps = 1e-12
            y_pred_max = torch.amax(y_pred, dim=(-1, -2, -3), keepdim=True) + eps
            y_pred = y_pred / y_pred_max

            # normalizing y
            y_max = torch.amax(y, dim=(-1, -2, -3), keepdim=True) + eps
            y = y / y_max

            # BEFORE
            # self.optimizer.zero_grad(set_to_none=True)

            # convert to CHW for loss and remove depth
            y_pred = y_pred.reshape(-1, *y_pred.shape[-3:]).movedim(-1, -3)
            y = y.reshape(-1, *y.shape[-3:]).movedim(-1, -3)

            # crop
            if self.crop is not None:
                y_pred = y_pred[
                    ...,
                    self.crop["vertical"][0] : self.crop["vertical"][1],
                    self.crop["horizontal"][0] : self.crop["horizontal"][1],
                ]
                y = y[
                    ...,
                    self.crop["vertical"][0] : self.crop["vertical"][1],
                    self.crop["horizontal"][0] : self.crop["horizontal"][1],
                ]

            loss_v = self.Loss(y_pred, y)
            if self.lpips:

                if y_pred.shape[1] == 1:
                    # if only one channel, repeat for LPIPS
                    y_pred = y_pred.repeat(1, 3, 1, 1)
                    y = y.repeat(1, 3, 1, 1)

                # value for LPIPS needs to be in range [-1, 1]
                loss_v = loss_v + self.lpips * torch.mean(
                    self.Loss_lpips(2 * y_pred - 1, 2 * y - 1)
                )
            if self.use_mask and self.l1_mask:
                for p in self.mask.parameters():
                    if p.requires_grad:
                        loss_v = loss_v + self.l1_mask * torch.mean(torch.abs(p))
            loss_v.backward()

            # check mask parameters are learning
            if self.use_mask:
                for p in self.mask.parameters():
                    assert p.grad is not None

            if self.clip_grad_norm is not None:
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

            # NEW
            self.optimizer.zero_grad(set_to_none=True)

            # update mask
            if self.use_mask:
                self.mask.update_mask()
                # NEW
                self.train_dataloader.dataset.set_psf()
                self.recon._set_psf(self.mask._psf)

            mean_loss += (loss_v.item() - mean_loss) * (1 / i)
            pbar.set_description(f"loss : {mean_loss}")
            i += 1

        self.print(f"loss : {mean_loss}")

        return mean_loss

    def evaluate(self, mean_loss, save_pt, epoch, disp=None):
        """
        Evaluate the reconstruction algorithm on the test dataset.

        Parameters
        ----------
        mean_loss : float
            Mean loss of the last epoch.
        save_pt : str
            Path to save metrics dictionary to. If None, no logging of metrics.
        disp : list of int, optional
            Test set examples to visualize at the end of each epoch, by default None.
        """
        if self.test_dataset is None:
            return

        # NEW
        if self.use_mask:
            with torch.no_grad():
                self.test_dataset.set_psf()

        output_dir = None
        if disp is not None:
            output_dir = os.path.join("eval_recon")
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_dir = os.path.join(output_dir, str(epoch))

        # benchmarking
        current_metrics = benchmark(
            self.recon,
            self.test_dataset,
            batchsize=self.eval_batch_size,
            save_idx=disp,
            output_dir=output_dir,
            crop=self.crop,
        )
        
        # update metrics with current metrics
        self.metrics["LOSS"].append(mean_loss)
        for key in current_metrics:
            self.metrics[key].append(current_metrics[key])

        if save_pt:
            # save dictionary metrics to file with json
            with open(os.path.join(save_pt, "metrics.json"), "w") as f:
                json.dump(self.metrics, f, indent=4)

        # check best metric
        if self.metrics["metric_for_best_model"] is None:
            eval_loss = current_metrics["MSE"]
            if self.lpips is not None:
                eval_loss += self.lpips * current_metrics["LPIPS_Vgg"]
            if self.use_mask and self.l1_mask:
                for p in self.mask.parameters():
                    if p.requires_grad:
                        eval_loss += self.l1_mask * np.mean(np.abs(p.cpu().detach().numpy()))
                # eval_loss += self.l1_mask * np.mean(np.abs(self.mask._mask.cpu().detach().numpy()))
            return eval_loss
        else:
            return current_metrics[self.metrics["metric_for_best_model"]]

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
        # self.save(path=save_pt, include_optimizer=False)
        epoch_eval_metric = self.evaluate(mean_loss, save_pt, epoch, disp=disp)
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
        #self.evaluate(-1, save_pt, epoch=0, disp=disp)
        
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
            self.scheduler.step()

        self.print(f"Train time : {time.time() - start_time} s")

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
            save_image(psf_np, os.path.join(path, f"psf_epoch{epoch}.png"))
            plot_image(psf_np, gamma=self.gamma)
            plt.savefig(os.path.join(path, f"psf_epoch{epoch}_plot.png"))

        # save optimizer
        if include_optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(path, f"optim_epoch{epoch}.pt"))

        # save recon
        torch.save(self.recon.state_dict(), os.path.join(path, f"recon_epoch{epoch}"))
