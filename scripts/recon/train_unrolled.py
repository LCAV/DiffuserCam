# #############################################################################
# train_unrolled.py
# =================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

"""
Train unrolled version of reconstruction algorithm.

```
python scripts/recon/train_unrolled.py
```

By default it uses the configuration from the file `configs/train_unrolledADMM.yaml`.

To train pre- and post-processing networks, use the following command:
```
python scripts/recon/train_unrolled.py -cn train_pre-post-processing
```

To fine-tune the DiffuserCam PSF, use the following command:
```
python scripts/recon/train_unrolled.py -cn fine-tune_PSF
```

To train a PSF from scratch with a simulated dataset, use the following command:
```
python scripts/recon/train_unrolled.py -cn train_psf_from_scratch
```

"""

import logging
import hydra
from hydra.utils import get_original_cwd
import os
import numpy as np
import time
from lensless.hardware.trainable_mask import prep_trainable_mask
from lensless import UnrolledFISTA, UnrolledADMM, TrainableInversion
from lensless.utils.dataset import (
    DiffuserCamMirflickr,
    DigiCamCelebA,
    DigiCam,
)
from torch.utils.data import Subset
from lensless.recon.utils import create_process_network
from lensless.utils.dataset import simulate_dataset
from lensless.recon.utils import Trainer
import torch
from lensless.utils.io import save_image
from lensless.utils.plot import plot_image
from lensless import ADMM
import matplotlib.pyplot as plt

# A logger for this file
log = logging.getLogger(__name__)


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@hydra.main(version_base=None, config_path="../../configs", config_name="train_unrolledADMM")
def train_unrolled(config):

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
    if "cuda" in config.torch_device and torch.cuda.is_available():
        # if config.torch_device == "cuda" and torch.cuda.is_available():
        log.info("Using GPU for training.")
        device = config.torch_device
        use_cuda = True
    else:
        log.info("Using CPU for training.")
        device = "cpu"
    # device, use_cuda, multi_gpu, device_ids = device_checks(
    #     config.torch_device, config.multi_gpu, logger=log.info,
    # )
    device_ids = config.device_ids

    # load dataset and create dataloader
    train_set = None
    test_set = None
    psf = None
    crop = None
    mask = None
    if "DiffuserCam" in config.files.dataset:

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
        if mask is not None:
            # plot initial PSF
            psf_np = mask.get_psf().detach().cpu().numpy()[0, ...]
            if config.trainable_mask.grayscale:
                psf_np = psf_np[:, :, -1]

            save_image(psf_np, os.path.join(save, "psf_initial.png"))
            plot_image(psf_np, gamma=config.display.gamma)
            plt.savefig(os.path.join(save, "psf_initial_plot.png"))

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
            # plot initial PSF
            with torch.no_grad():
                psf_np = mask.get_psf().detach().cpu().numpy()[0, ...]
                if config.trainable_mask.grayscale:
                    psf_np = psf_np[:, :, -1]

            save_image(psf_np, os.path.join(save, "psf_initial.png"))
            plot_image(psf_np, gamma=config.display.gamma)
            plt.savefig(os.path.join(save, "psf_initial_plot.png"))

            # save original PSF as well
            psf_meas = dataset.psf.detach().cpu().numpy()[0, ...]
            plot_image(psf_meas, gamma=config.display.gamma)
            plt.savefig(os.path.join(save, "psf_meas_plot.png"))

            with torch.no_grad():
                psf = mask.get_psf().to(dataset.psf)

        else:

            psf = dataset.psf

        # print info about PSF
        log.info(f"PSF shape : {psf.shape}")
        log.info(f"PSF min : {psf.min()}")
        log.info(f"PSF max : {psf.max()}")
        log.info(f"PSF dtype : {psf.dtype}")
        log.info(f"PSF norm : {psf.norm()}")

    elif config.files.huggingface_dataset is True:

        train_set = DigiCam(
            huggingface_repo=config.files.dataset,
            split="train",
            display_res=config.files.image_res,
            rotate=config.files.rotate,
            downsample=config.files.downsample,
            alignment=config.alignment,
            save_psf=config.files.save_psf,
        )
        test_set = DigiCam(
            huggingface_repo=config.files.dataset,
            split="test",
            display_res=config.files.image_res,
            rotate=config.files.rotate,
            downsample=config.files.downsample,
            alignment=config.alignment,
            save_psf=config.files.save_psf,
        )
        if train_set.multimask:
            # get first PSF for initialization
            first_psf_key = list(train_set.psf.keys())[device_ids[0]]
            psf = train_set.psf[first_psf_key].to(device)
        else:
            psf = train_set.psf.to(device)

    else:

        train_set, test_set, mask = simulate_dataset(config, generator=generator)
        psf = train_set.psf
        crop = train_set.crop

    assert train_set is not None
    # if not hasattr(test_set, "psfs"):
    #     assert psf is not None

    if config.files.extra_eval is not None:
        # TODO only support Hugging Face DigiCam datasets for now
        extra_eval_sets = dict()
        for eval_set in config.files.extra_eval:

            extra_eval_sets[eval_set] = DigiCam(
                split="test",
                downsample=config.files.downsample,  # needs to be same size
                **config.files.extra_eval[eval_set],
            )

    # reconstruct lensless with ADMM
    with torch.no_grad():
        if config.eval_disp_idx is not None:

            log.info("Reconstruction a few images with ADMM...")

            for i, _idx in enumerate(config.eval_disp_idx):

                if test_set.multimask:
                    # multimask
                    # lensless, lensed, _ = test_set[_idx]  # using wrong PSF
                    lensless, lensed, psf = test_set[_idx]
                    psf = psf.to(device)
                else:
                    lensless, lensed = test_set[_idx]
                recon = ADMM(psf)

                recon.set_data(lensless.to(psf.device))
                res = recon.apply(disp_iter=None, plot=False, n_iter=10)
                res_np = res[0].cpu().numpy()
                res_np = res_np / res_np.max()

                lensed_np = lensed[0].cpu().numpy()

                lensless_np = lensless[0].cpu().numpy()
                save_image(lensless_np, f"lensless_raw_{_idx}.png")

                # -- plot lensed and res on top of each other
                cropped = False
                if test_set.alignment is not None:
                    top_right = test_set.alignment["topright"]
                    height = test_set.alignment["height"]
                    width = test_set.alignment["width"]
                    res_np = res_np[
                        top_right[0] : top_right[0] + height, top_right[1] : top_right[1] + width
                    ]
                    cropped = True

                elif config.training.crop_preloss:
                    assert crop is not None

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

                save_image(res_np, f"lensless_recon_{_idx}.png")
                save_image(lensed_np, f"lensed_{_idx}.png")

                plt.figure()
                plt.imshow(lensed_np, alpha=0.4)
                plt.imshow(res_np, alpha=0.7)
                plt.savefig(f"overlay_lensed_recon_{_idx}.png")

    log.info(f"Train test size : {len(train_set)}")
    log.info(f"Test test size : {len(test_set)}")

    start_time = time.time()

    # Load pre-process model
    pre_process, pre_process_name = create_process_network(
        config.reconstruction.pre_process.network,
        config.reconstruction.pre_process.depth,
        nc=config.reconstruction.pre_process.nc,
        device=device,
        device_ids=device_ids,
    )
    pre_proc_delay = config.reconstruction.pre_process.delay

    # Load post-process model
    post_process, post_process_name = create_process_network(
        config.reconstruction.post_process.network,
        config.reconstruction.post_process.depth,
        nc=config.reconstruction.post_process.nc,
        device=device,
        device_ids=device_ids,
    )
    post_proc_delay = config.reconstruction.post_process.delay

    if config.reconstruction.post_process.train_last_layer:
        for name, param in post_process.named_parameters():
            if "m_tail" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            # print(name, param.requires_grad, param.numel())

    # initialize pre- and post processor with another model
    if config.reconstruction.init_processors is not None:
        from lensless.recon.model_dict import load_model, model_dict

        model_orig = load_model(
            model_dict["diffusercam"]["mirflickr"][config.reconstruction.init_processors],
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

    # create reconstruction algorithm
    if config.reconstruction.method == "unrolled_fista":
        recon = UnrolledFISTA(
            psf,
            n_iter=config.reconstruction.unrolled_fista.n_iter,
            tk=config.reconstruction.unrolled_fista.tk,
            pad=True,
            learn_tk=config.reconstruction.unrolled_fista.learn_tk,
            pre_process=pre_process if pre_proc_delay is None else None,
            post_process=post_process if post_proc_delay is None else None,
            skip_unrolled=config.reconstruction.skip_unrolled,
            return_unrolled_output=True if config.unrolled_output_factor > 0 else False,
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
            skip_unrolled=config.reconstruction.skip_unrolled,
            return_unrolled_output=True if config.unrolled_output_factor > 0 else False,
        )
    elif config.reconstruction.method == "trainable_inv":
        recon = TrainableInversion(
            psf,
            K=config.reconstruction.trainable_inv.K,
            pre_process=pre_process if pre_proc_delay is None else None,
            post_process=post_process if post_proc_delay is None else None,
            return_unrolled_output=True if config.unrolled_output_factor > 0 else False,
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
        extra_eval_sets=extra_eval_sets if config.files.extra_eval is not None else None,
    )

    trainer.train(n_epoch=config.training.epoch, save_pt=save, disp=config.eval_disp_idx)

    log.info(f"Results saved in {save}")


if __name__ == "__main__":
    train_unrolled()
