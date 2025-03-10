# #############################################################################
# train_learning_based.py
# =======================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Base script for training learned reconstruction approaches.

Check out this page for configuration details:
https://github.com/LCAV/LenslessPiCam/tree/main/configs/train#training-physics-informed-reconstruction-models


"""

import wandb
import logging
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import os
import numpy as np
import time
from lensless.utils.image import shift_with_pad
from lensless.hardware.trainable_mask import prep_trainable_mask
from lensless import ADMM, UnrolledFISTA, UnrolledADMM, TrainableInversion, SVDeconvNet
from lensless.recon.multi_wiener import MultiWiener
from lensless.recon.integrated_background_sub import IntegratedBackgroundSub
from lensless.utils.dataset import (
    DiffuserCamMirflickr,
    DigiCamCelebA,
    HFDataset,
    MyDataParallel,
    simulate_dataset,
    HFSimulated,
)
from torch.utils.data import Subset
from lensless.recon.utils import create_process_network
from lensless.recon.utils import Trainer
import torch
from lensless.utils.io import save_image
from lensless.utils.plot import plot_image
import matplotlib.pyplot as plt
from lensless.recon.model_dict import load_model, download_model

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs/train", config_name="defaults")
def train_learned(config):

    if config.wandb_project is not None:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.wandb_project,
            # track hyperparameters and run metadata
            config=dict(config),
        )

    # set seed
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    if config.start_delay is not None:
        # wait for this time before starting script
        delay = config.start_delay * 60
        start_time = time.time() + delay
        start_time = time.strftime("%H:%M:%S", time.localtime(start_time))
        print(f"\nScript will start at {start_time}")
        time.sleep(delay)

    save = config.save
    if save:
        save = os.getcwd()

    use_cuda = False

    # check torch_device is iterable
    if hasattr(config.torch_device, "__iter__") and not isinstance(config.torch_device, str):
        # assert every element is integer
        assert all(isinstance(x, int) for x in config.torch_device)
        device = f"cuda:{config.torch_device[0]}"
        device_ids = config.torch_device
    else:
        device = config.torch_device
        device_ids = None

    if "cuda" in device and torch.cuda.is_available():
        log.info(f"Using GPU for training. Main device : {device}")
        use_cuda = True
    else:
        log.info("Using CPU for training.")
        device = "cpu"
    if device_ids is not None:
        log.info(f"Using multiple GPUs : {device_ids}")

    # load dataset and create dataloader
    train_set = None
    test_set = None
    psf = None
    crop = None
    mask = None
    if "DiffuserCam" in config.files.dataset and config.files.huggingface_dataset is False:

        original_path = os.path.join(get_original_cwd(), config.files.dataset)
        psf_path = os.path.join(get_original_cwd(), config.files.psf)
        dataset = DiffuserCamMirflickr(
            dataset_dir=original_path,
            psf_path=psf_path,
            downsample=config.files.downsample,
            input_snr=config.files.input_snr,
        )
        dataset.psf = dataset.psf.to(device)
        # train-test split as in https://waller-lab.github.io/LenslessLearning/dataset.html
        # first 1000 files for test, the rest for training
        train_indices = dataset.allowed_idx[dataset.allowed_idx > 1000]
        test_indices = dataset.allowed_idx[dataset.allowed_idx <= 1000]
        if config.files.n_files is not None:
            train_indices = train_indices[: config.files.n_files]
            test_indices = test_indices[: config.files.n_files]

        train_set = Subset(dataset, train_indices)
        test_set = Subset(dataset, test_indices)

        # -- if learning mask
        mask = prep_trainable_mask(config, dataset.psf)

        psf = dataset.psf

    elif "celeba_adafruit" in config.files.dataset:

        dataset = DigiCamCelebA(
            data_dir=os.path.join(get_original_cwd(), config.files.dataset),
            celeba_root=config.files.celeba_root,
            psf_path=os.path.join(get_original_cwd(), config.files.psf),
            downsample=config.files.downsample,
            vertical_shift=config.files.vertical_shift,
            horizontal_shift=config.files.horizontal_shift,
            simulation_config=config.simulation,
            crop=config.files.crop,
            input_snr=config.files.input_snr,
        )
        crop = dataset.crop
        dataset.psf = dataset.psf.to(device)
        log.info(f"Data shape :  {dataset[0][0].shape}")

        if config.files.n_files is not None:
            dataset = Subset(dataset, np.arange(config.files.n_files))
            dataset.psf = dataset.dataset.psf

        # train-test split
        train_size = int((1 - config.files.test_size) * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, test_size], generator=generator
        )

        # -- if learning mask
        downsample = config.files.downsample * 4  # measured files are 4x downsampled
        mask = prep_trainable_mask(config, dataset.psf, downsample=downsample)

        if mask is not None:
            # save original PSF
            psf_meas = dataset.psf.detach().cpu().numpy()[0, ...]
            plot_image(psf_meas, gamma=config.display.gamma)
            plt.savefig(os.path.join(save, "psf_meas_plot.png"))

            with torch.no_grad():
                psf = mask.get_psf().to(dataset.psf)

        else:

            psf = dataset.psf

    elif config.files.huggingface_dataset is True:

        split_train = "train"
        split_test = "test"
        if config.files.split_seed is not None:
            from datasets import load_dataset, concatenate_datasets

            seed = config.files.split_seed
            generator = torch.Generator().manual_seed(seed)

            # - combine train and test into single dataset
            train_split = "train"
            test_split = "test"
            if config.files.n_files is not None:
                train_split = f"train[:{config.files.n_files}]"
                test_split = f"test[:{config.files.n_files}]"
            train_dataset = load_dataset(
                config.files.dataset, split=train_split, cache_dir=config.files.cache_dir
            )
            test_dataset = load_dataset(
                config.files.dataset, split=test_split, cache_dir=config.files.cache_dir
            )
            dataset = concatenate_datasets([test_dataset, train_dataset])

            # - split into train and test
            train_size = int((1 - config.files.test_size) * len(dataset))
            test_size = len(dataset) - train_size
            split_train, split_test = torch.utils.data.random_split(
                dataset, [train_size, test_size], generator=generator
            )

        if config.files.hf_simulated:
            # simulate lensless by using measured PSF
            train_set = HFSimulated(
                huggingface_repo=config.files.dataset,
                split=split_train,
                n_files=config.files.n_files,
                psf=config.files.huggingface_psf,
                downsample=config.files.downsample,
                cache_dir=config.files.cache_dir,
                single_channel_psf=config.files.single_channel_psf,
                flipud=config.files.flipud,
                display_res=config.files.image_res,
                alignment=config.alignment,
                bg_snr_range=config.files.background_snr_range,  # TODO check if correct
                bg_fp=(
                    to_absolute_path(config.files.background_fp)
                    if config.files.background_fp is not None
                    else None
                ),
                input_snr=config.files.input_snr,
            )

        else:
            train_set = HFDataset(
                huggingface_repo=config.files.dataset,
                cache_dir=config.files.cache_dir,
                psf=config.files.huggingface_psf,
                single_channel_psf=config.files.single_channel_psf,
                split=split_train,
                display_res=config.files.image_res,
                rotate=config.files.rotate,
                flipud=config.files.flipud,
                flip_lensed=config.files.flip_lensed,
                downsample=config.files.downsample,
                downsample_lensed=config.files.downsample_lensed,
                alignment=config.alignment,
                save_psf=config.files.save_psf,
                n_files=config.files.n_files,
                simulation_config=config.simulation,
                force_rgb=config.files.force_rgb,
                simulate_lensless=config.files.simulate_lensless,
                random_flip=config.files.random_flip,
                bg_snr_range=config.files.background_snr_range,
                bg_fp=(
                    to_absolute_path(config.files.background_fp)
                    if config.files.background_fp is not None
                    else None
                ),
                input_snr=config.files.input_snr,
                psf_snr=config.files.psf_snr,
            )

        test_set = HFDataset(
            huggingface_repo=config.files.dataset,
            cache_dir=config.files.cache_dir,
            psf=config.files.huggingface_psf,
            single_channel_psf=config.files.single_channel_psf,
            split=split_test,
            display_res=config.files.image_res,
            rotate=config.files.rotate,
            flipud=config.files.flipud,
            flip_lensed=config.files.flip_lensed,
            downsample=config.files.downsample,
            downsample_lensed=config.files.downsample_lensed,
            alignment=config.alignment,
            save_psf=config.files.save_psf,
            n_files=config.files.n_files,
            simulation_config=config.simulation,
            bg_snr_range=config.files.background_snr_range,
            bg_fp=(
                to_absolute_path(config.files.background_fp)
                if config.files.background_fp is not None
                else None
            ),
            force_rgb=config.files.force_rgb,
            simulate_lensless=False,  # in general evaluate on measured (set to False)
            input_snr=config.files.input_snr,
        )

        if config.files.psf_snr is not None:
            # overwrite test set PSF with train set PSF
            test_set.psf = train_set.psf

        if train_set.multimask:
            # get first PSF for initialization
            if device_ids is not None:
                first_psf_key = list(train_set.psf.keys())[device_ids[0]]
            else:
                first_psf_key = list(train_set.psf.keys())[0]
            psf = train_set.psf[first_psf_key].to(device)
        else:
            psf = train_set.psf.to(device)
        crop = test_set.crop  # same for train set

        # -- if learning mask
        mask = prep_trainable_mask(config, psf)
        if mask is not None:
            assert not train_set.multimask

    else:

        train_set, test_set, mask = simulate_dataset(config, generator=generator)
        psf = train_set.psf
        crop = train_set.crop

    if not hasattr(train_set, "multimask"):
        train_set.multimask = False
    if not hasattr(test_set, "multimask"):
        test_set.multimask = False

    assert train_set is not None
    # if not hasattr(test_set, "psfs"):
    #     assert psf is not None

    # print info about PSF
    log.info(f"PSF shape : {psf.shape}")
    log.info(f"PSF min : {psf.min()}")
    log.info(f"PSF max : {psf.max()}")
    log.info(f"PSF dtype : {psf.dtype}")
    log.info(f"PSF norm : {psf.norm()}")

    if config.files.extra_eval is not None:
        # TODO only support Hugging Face DigiCam datasets for now
        extra_eval_sets = dict()
        for eval_set in config.files.extra_eval:

            extra_eval_sets[eval_set] = HFDataset(
                split="test",
                downsample=config.files.downsample,  # needs to be same size
                n_files=config.files.n_files,
                simulation_config=config.simulation,
                simulate_lensless=False,  # in general evaluate on measured
                **config.files.extra_eval[eval_set],
            )

    # reconstruct lensless with ADMM
    with torch.no_grad():
        if config.eval_disp_idx is not None:

            log.info("Reconstruction a few images with ADMM...")

            for i, _idx in enumerate(config.eval_disp_idx):

                flip_lr = None
                flip_ud = None
                return_items = test_set[_idx]
                lensless = return_items[0]
                lensed = return_items[1]
                if test_set.bg_sim is not None or test_set.measured_bg:
                    background = return_items[-1]
                if test_set.multimask or test_set.random_flip:
                    psf_recon = return_items[2]
                    psf_recon = psf_recon.to(device)
                else:
                    psf_recon = psf.clone()
                if test_set.random_flip:
                    flip_lr = return_items[3]
                    flip_ud = return_items[4]

                rotate_angle = False
                if config.files.random_rotate:
                    from lensless.utils.image import rotate_HWC

                    rotate_angle = np.random.uniform(
                        -config.files.random_rotate, config.files.random_rotate
                    )
                    print(f"Rotate angle : {rotate_angle}")
                    lensless = rotate_HWC(lensless, rotate_angle)
                    lensed = rotate_HWC(lensed, rotate_angle)
                    psf_recon = rotate_HWC(psf_recon, rotate_angle)

                shift = None
                if config.files.random_shifts:

                    shift = np.random.randint(
                        -config.files.random_shifts, config.files.random_shifts, 2
                    )
                    print(f"Shift : {shift}")
                    lensless = shift_with_pad(lensless, shift, axis=(1, 2))
                    lensed = shift_with_pad(lensed, shift, axis=(1, 2))
                    psf_recon = shift_with_pad(psf_recon, shift, axis=(1, 2))
                    shift = tuple(shift)

                if config.files.random_rotate or config.files.random_shifts:
                    save_image(psf_recon[0].cpu().numpy(), f"psf_{_idx}.png")

                save_image(lensed[0].cpu().numpy(), f"lensed_{_idx}.png")
                save_image(lensless[0].cpu().numpy(), f"lensless_raw_{_idx}.png")

                # Reconstruct and plot image
                reconstruct_save(
                    _idx,
                    config,
                    crop,
                    i,
                    lensed,
                    lensless,
                    psf_recon,
                    test_set,
                    "",
                    flip_lr,
                    flip_ud,
                    rotate_angle,
                    shift,
                )
                # save_image(lensed, f"lensed_{_idx}.png")
                if test_set.bg_sim is not None or test_set.measured_bg:
                    # Reconstruct and plot background subtracted image
                    reconstruct_save(
                        _idx,
                        config,
                        crop,
                        i,
                        lensed,
                        (lensless - background),
                        psf_recon,
                        test_set,
                        "subtraction_",
                        flip_lr,
                        flip_ud,
                        rotate_angle,
                        shift,
                    )
    log.info(f"Train test size : {len(train_set)}")
    log.info(f"Test test size : {len(test_set)}")

    start_time = time.time()

    # Load pre-process model
    pre_proc_delay = config.reconstruction.pre_process.delay
    pre_process, pre_process_name = create_process_network(
        config.reconstruction.pre_process.network,
        device=device,
        device_ids=device_ids,
        background_subtraction=config.reconstruction.integrated_background_unetres,
        input_background=config.reconstruction.unetres_input_background,
        # unet param
        depth=config.reconstruction.pre_process.depth,
        nc=config.reconstruction.pre_process.nc,
        # restormer param
        restormer_params=config.reconstruction.pre_process.restormer_params,
    )

    # Load post-process model
    post_process, post_process_name = create_process_network(
        config.reconstruction.post_process.network,
        device=device,
        device_ids=device_ids,
        concatenate_compensation=(
            config.reconstruction.compensation[-1]
            if config.reconstruction.compensation is not None
            else False
        ),
        # unet param
        depth=config.reconstruction.post_process.depth,
        nc=config.reconstruction.post_process.nc,
        # restormer param
        restormer_params=config.reconstruction.post_process.restormer_params,
    )
    post_proc_delay = config.reconstruction.post_process.delay

    if post_process is not None and config.reconstruction.post_process.train_last_layer:
        for name, param in post_process.named_parameters():
            if "m_tail" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            # print(name, param.requires_grad, param.numel())

    # initialize pre- and post processor with another model
    if config.reconstruction.init_processors is not None:

        if "hf" in config.reconstruction.init_processors:
            param = config.reconstruction.init_processors.split(":")
            camera = param[1]
            dataset = param[2]
            model_name = param[3]
            model_path = download_model(camera=camera, dataset=dataset, model=model_name)

        elif "local" in config.reconstruction.init_processors:
            model_path = config.reconstruction.init_processors.split(":")[1]

        else:
            raise ValueError(f"{config.reconstruction.init_processors} is not a supported model")

        model_orig = load_model(
            model_path=model_path,
            psf=psf,
            device=device,
        )

        # -- replace pre-process
        if config.reconstruction.init_pre:
            params1 = model_orig.pre_process_model.named_parameters()
            params2 = pre_process.named_parameters()
            dict_params2 = dict(params2)
            for name1, param1 in params1:
                if name1 in dict_params2:
                    dict_params2[name1].data.copy_(param1.data)

        # -- replace post-process
        if config.reconstruction.init_post:
            params1_post = model_orig.post_process_model.named_parameters()
            params2_post = post_process.named_parameters()
            dict_params2_post = dict(params2_post)
            for name1, param1 in params1_post:
                if name1 in dict_params2_post:
                    dict_params2_post[name1].data.copy_(param1.data)

    # network for PSF
    psf_network = None
    if config.reconstruction.psf_network:
        # create UnetRes for PSF
        psf_network, _ = create_process_network(
            network="UnetRes",
            depth=len(config.reconstruction.psf_network),
            nc=config.reconstruction.psf_network,
            device=device,
            device_ids=device_ids,
        )

    # check/prepare background subtraction
    background_network = None
    if config.reconstruction.direct_background_subtraction:
        assert test_set.measured_bg and train_set.measured_bg
        assert config.reconstruction.learned_background_subtraction is False
    elif config.reconstruction.learned_background_subtraction:
        assert config.reconstruction.direct_background_subtraction is False
        assert test_set.measured_bg and train_set.measured_bg

        # create UnetRes for background subtraction
        background_network, _ = create_process_network(
            network="UnetRes",
            depth=len(config.reconstruction.learned_background_subtraction),
            nc=config.reconstruction.learned_background_subtraction,
            device=device,
            device_ids=device_ids,
        )
    elif config.reconstruction.integrated_background_subtraction:
        raise NotImplementedError("Integrated background subtraction not fully integrated yet")

        assert config.reconstruction.direct_background_subtraction is False
        assert config.reconstruction.learned_background_subtraction is False
        assert pre_process is None
        assert test_set.measured_bg and train_set.measured_bg

        background = train_set[0][-1]
        background_network = IntegratedBackgroundSub(
            in_channels=3,
            out_channels=3,
            input_shape=background.shape,
            nc=config.reconstruction.integrated_background_subtraction,
            down_subtraction=config.reconstruction.down_subtraction,
        )

    # create reconstruction algorithm
    if config.reconstruction.init is not None:
        assert config.reconstruction.init_processors is None

        param = config.reconstruction.init.split(":")
        assert len(param) == 4, "hf model requires following format: hf:camera:dataset:model_name"
        camera = param[1]
        dataset = param[2]
        model_name = param[3]
        model_path = download_model(camera=camera, dataset=dataset, model=model_name)
        recon = load_model(
            model_path,
            psf,
            device,
            device_ids=device_ids,
            train_last_layer=config.reconstruction.post_process.train_last_layer,
        )

    else:

        if config.reconstruction.method == "unrolled_fista":
            recon = UnrolledFISTA(
                psf,
                n_iter=config.reconstruction.unrolled_fista.n_iter,
                tk=config.reconstruction.unrolled_fista.tk,
                pad=True,
                learn_tk=config.reconstruction.unrolled_fista.learn_tk,
                pre_process=pre_process if pre_proc_delay is None else None,
                post_process=post_process if post_proc_delay is None else None,
                background_network=background_network,
                psf_network=psf_network,
                psf_residual=config.reconstruction.psf_residual,
                skip_unrolled=config.reconstruction.skip_unrolled,
                return_intermediate=(
                    True if config.unrolled_output_factor > 0 or config.pre_proc_aux > 0 else False
                ),
                compensation=config.reconstruction.compensation,
                compensation_residual=config.reconstruction.compensation_residual,
                direct_background_subtraction=config.reconstruction.direct_background_subtraction,
                integrated_background_subtraction=config.reconstruction.integrated_background_subtraction,
            )
        elif config.reconstruction.method == "unrolled_admm":
            recon = UnrolledADMM(
                psf,
                n_iter=config.reconstruction.unrolled_admm.n_iter,
                mu1=config.reconstruction.unrolled_admm.mu1,
                mu2=config.reconstruction.unrolled_admm.mu2,
                mu3=config.reconstruction.unrolled_admm.mu3,
                tau=config.reconstruction.unrolled_admm.tau,
                pre_process=pre_process if pre_proc_delay is None else None,
                post_process=post_process if post_proc_delay is None else None,
                background_network=background_network,
                psf_network=psf_network,
                psf_residual=config.reconstruction.psf_residual,
                skip_unrolled=config.reconstruction.skip_unrolled,
                return_intermediate=(
                    True if config.unrolled_output_factor > 0 or config.pre_proc_aux > 0 else False
                ),
                compensation=config.reconstruction.compensation,
                compensation_residual=config.reconstruction.compensation_residual,
                direct_background_subtraction=config.reconstruction.direct_background_subtraction,
                integrated_background_subtraction=config.reconstruction.integrated_background_subtraction,
            )
        elif config.reconstruction.method == "trainable_inv":
            assert config.trainable_mask.mask_type == "TrainablePSF"
            assert psf_network is None
            recon = TrainableInversion(
                psf,
                K=config.reconstruction.trainable_inv.K,
                pre_process=pre_process if pre_proc_delay is None else None,
                post_process=post_process if post_proc_delay is None else None,
                background_network=background_network,
                return_intermediate=(
                    True if config.unrolled_output_factor > 0 or config.pre_proc_aux > 0 else False
                ),
                direct_background_subtraction=config.reconstruction.direct_background_subtraction,
                integrated_background_subtraction=config.reconstruction.integrated_background_subtraction,
            )
        elif config.reconstruction.method == "svdeconvnet":
            assert config.trainable_mask.mask_type == "TrainablePSF"
            assert psf_network is None
            recon = SVDeconvNet(
                psf,
                K=config.reconstruction.svdeconvnet.K,
                pre_process=pre_process if pre_proc_delay is None else None,
                post_process=post_process if post_proc_delay is None else None,
                background_network=background_network,
                return_intermediate=(
                    True if config.unrolled_output_factor > 0 or config.pre_proc_aux > 0 else False
                ),
                direct_background_subtraction=config.reconstruction.direct_background_subtraction,
                integrated_background_subtraction=config.reconstruction.integrated_background_subtraction,
            )
        elif config.reconstruction.method == "multi_wiener":

            if config.files.single_channel_psf:
                psf = psf[..., 0].unsqueeze(-1)
                psf_channels = 1
            else:
                psf_channels = 3

            assert config.reconstruction.direct_background_subtraction is False, "Not supported"
            assert config.reconstruction.learned_background_subtraction is False, "Not supported"
            assert config.reconstruction.integrated_background_subtraction is False, "Not supported"
            assert psf_network is None, "Not supported"

            recon = MultiWiener(
                in_channels=3,
                out_channels=3,
                psf=psf,
                psf_channels=psf_channels,
                nc=config.reconstruction.multi_wiener.nc,
                pre_process=pre_process if pre_proc_delay is None else None,
            )

        else:
            raise ValueError(f"{config.reconstruction.method} is not a supported algorithm")

        if device_ids is not None:
            recon = MyDataParallel(recon, device_ids=device_ids)
        if use_cuda:
            recon.to(device)

    # constructing algorithm name by appending pre and post process
    algorithm_name = config.reconstruction.method
    if config.reconstruction.pre_process.network is not None:
        algorithm_name = pre_process_name + "_" + algorithm_name
    if config.reconstruction.post_process.network is not None:
        algorithm_name += "_" + post_process_name

    # print number of trainable parameters
    n_param = sum(p.numel() for p in recon.parameters() if p.requires_grad)
    if mask is not None:
        n_param += sum(p.numel() for p in mask.parameters() if p.requires_grad)
    log.info(f"Training model with {n_param} parameters")
    if pre_process is not None:
        n_param = sum(p.numel() for p in pre_process.parameters() if p.requires_grad)
        log.info(f"-- Pre-process model with {n_param} parameters")
    if post_process is not None:
        n_param = sum(p.numel() for p in post_process.parameters() if p.requires_grad)
        log.info(f"-- Post-process model with {n_param} parameters")
    if background_network is not None:
        n_param = sum(p.numel() for p in background_network.parameters() if p.requires_grad)
        log.info(f"-- Background subtraction model with {n_param} parameters")
    if psf_network is not None:
        n_param = sum(p.numel() for p in psf_network.parameters() if p.requires_grad)
        log.info(f"-- PSF network model with {n_param} parameters")
    if mask is not None:
        n_param = sum(p.numel() for p in mask.parameters() if p.requires_grad)
        log.info(f"-- Trainable mask with {n_param} parameters")

    log.info(f"Setup time : {time.time() - start_time} s")
    log.info(f"PSF shape : {psf.shape}")
    log.info(f"Results saved in {save}")
    trainer = Trainer(
        recon=recon,
        train_dataset=train_set,
        test_dataset=test_set,
        mask=mask,
        batch_size=config.training.batch_size,
        eval_batch_size=config.training.eval_batch_size,
        loss=config.loss,
        lpips=config.lpips,
        l1_mask=config.trainable_mask.L1_strength,
        optimizer=config.optimizer,
        skip_NAN=config.training.skip_NAN,
        algorithm_name=algorithm_name,
        metric_for_best_model=config.training.metric_for_best_model,
        save_every=config.training.save_every,
        gamma=config.display.gamma,
        logger=log,
        crop=crop if config.training.crop_preloss else None,
        pre_process=pre_process,
        pre_process_delay=pre_proc_delay,
        pre_process_freeze=config.reconstruction.pre_process.freeze,
        pre_process_unfreeze=config.reconstruction.pre_process.unfreeze,
        post_process=post_process,
        post_process_delay=post_proc_delay,
        post_process_freeze=config.reconstruction.post_process.freeze,
        post_process_unfreeze=config.reconstruction.post_process.unfreeze,
        clip_grad=config.training.clip_grad,
        unrolled_output_factor=config.unrolled_output_factor,
        pre_proc_aux=config.pre_proc_aux,
        extra_eval_sets=extra_eval_sets if config.files.extra_eval is not None else None,
        use_wandb=True if config.wandb_project is not None else False,
        n_epoch=config.training.epoch,
        random_rotate=config.files.random_rotate,
        random_shift=config.files.random_shifts,
    )

    trainer.train(n_epoch=config.training.epoch, save_pt=save, disp=config.eval_disp_idx)

    log.info(f"Results saved in {save}")


def reconstruct_save(
    _idx,
    config,
    crop,
    i,
    lensed,
    lensless,
    psf_recon,
    test_set,
    fp,
    flip_lr,
    flip_ud,
    rotate_angle,
    shift,
):
    recon = ADMM(psf_recon)

    recon.set_data(lensless.to(psf_recon.device))
    # recon.set_data(torch.from_numpy(lensless).to(psf_recon.device))
    res = recon.apply(disp_iter=None, plot=False, n_iter=10)
    res_np = res[0].cpu().numpy()
    res_np = res_np / res_np.max()
    lensed_np = lensed[0]  # .cpu().numpy()

    # -- plot lensed and res on top of each other
    cropped = False
    if hasattr(test_set, "alignment"):
        if test_set.alignment is not None:
            res_np = test_set.extract_roi(
                res_np,
                axis=(0, 1),
                flip_lr=flip_lr,
                flip_ud=flip_ud,
                rotate_aug=rotate_angle,
                shift_aug=shift,
            )
        else:
            res_np, lensed_np = test_set.extract_roi(
                res_np,
                lensed=lensed_np,
                axis=(0, 1),
                flip_lr=flip_lr,
                flip_ud=flip_ud,
                rotate_aug=rotate_angle,
                shift_aug=shift,
            )
        cropped = True

    elif config.training.crop_preloss:
        assert crop is not None
        assert flip_lr is None and flip_ud is None

        res_np = res_np[
            crop["vertical"][0] : crop["vertical"][1],
            crop["horizontal"][0] : crop["horizontal"][1],
        ]
        lensed_np = lensed_np[
            crop["vertical"][0] : crop["vertical"][1],
            crop["horizontal"][0] : crop["horizontal"][1],
        ]
        cropped = True

    if cropped and i == 0:
        log.info(f"Cropped shape :  {res_np.shape}")

    save_image(res_np, f"lensless_recon_{fp}{_idx}.png")

    plt.figure()
    plt.imshow(lensed_np, alpha=0.4)
    plt.imshow(res_np, alpha=0.7)
    plt.savefig(f"overlay_lensed_recon_{fp}{_idx}.png")


if __name__ == "__main__":
    train_learned()
