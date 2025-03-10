"""
First key is camera, second key is training data, third key is model name.

Download link corresponds to output folder from training
script, which contains the model checkpoint and config file,
and other intermediate files. Models are stored on Hugging Face.
"""


import os
from lensless.recon.integrated_background_sub import IntegratedBackgroundSub
import numpy as np
import torch
from lensless.recon.utils import create_process_network
from lensless.recon.unrolled_admm import UnrolledADMM
from lensless.recon.trainable_inversion import TrainableInversion
from lensless.hardware.trainable_mask import prep_trainable_mask
import yaml
from lensless.recon.multi_wiener import MultiWiener
from lensless import SVDeconvNet
from huggingface_hub import snapshot_download
from collections import OrderedDict
from lensless.utils.dataset import MyDataParallel


model_dir_path = os.path.join(os.path.dirname(__file__), "..", "..", "models")

model_dict = {
    "diffusercam": {
        "mirflickr": {
            # -- only unrolled20
            "U20": "bezzam/diffusercam-mirflickr-unrolled-admm20",
            "U20_0db": "bezzam/diffusercam-mirflickr-unrolled-admm20-0db",
            "U20_10db": "bezzam/diffusercam-mirflickr-unrolled-admm20-10db",
            "U20_20db": "bezzam/diffusercam-mirflickr-unrolled-admm20-20db",
            # -- only pre-process
            "Unet+U20": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20",
            "Unet+U20_0dB": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-0db",
            "Unet+U20_10db": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-10db",
            "Unet+U20_20db": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-20db",
            # -- only post-process
            "U20+Unet": "bezzam/diffusercam-mirflickr-unrolled-admm20-unet2",
            "U20+Unet_0db": "bezzam/diffusercam-mirflickr-unrolled-admm20-unet2-0db",
            "U20+Unet_10db": "bezzam/diffusercam-mirflickr-unrolled-admm20-unet2-10db",
            "U20+Unet_20db": "bezzam/diffusercam-mirflickr-unrolled-admm20-unet2-20db",
            # -- only post-process (Drunet)
            "U20+Drunet": "bezzam/diffusercam-mirflickr-unrolled-admm20-drunet",
            "TrainInv+Drunet": "bezzam/diffusercam-mirflickr-trainable-inv-drunet",
            # -- both
            "Unet+TrainInv+Unet": "bezzam/diffusercam-mirflickr-unet2-trainable-inv-unet2",
            "Unet+U20+Unet": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2",  # init with 0.01 and only pre-proc
            "Unet+U20+Unet_aux0.01": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2-aux0.01",
            "Unet+U20+Unet_aux0.03": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2-aux0.03",
            "Unet+U20+Unet_aux0.1": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2-aux0.1",  # init with 0.01
            "Unet+U20+Unet_aux1": "bezzam/diffusercam-mirflickr-unet2-unrolled-admm20-unet2-aux1",  # init with 0.01 and only pre-proc
            # baseline benchmarks which don't have model file but use ADMM
            "admm_fista": "bezzam/diffusercam-mirflickr-admm-fista",
            "admm_pnp": "bezzam/diffusercam-mirflickr-admm-pnp",
            # -- TCI submission
            "TrainInv+Unet8M": "bezzam/diffusercam-mirflickr-trainable-inv-unet8M",
            "Unet4M+U5+Unet4M": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M",
            "MWDN8M": "bezzam/diffusercam-mirflickr-mwdn-8M",
            "Unet2M+MWDN6M": "bezzam/diffusercam-mirflickr-unet2M-mwdn-6M",
            "Unet4M+TrainInv+Unet4M": "bezzam/diffusercam-mirflickr-unet4M-trainable-inv-unet4M",
            "MMCN4M+Unet4M": "bezzam/diffusercam-mirflickr-mmcn-unet4M",
            "U5+Unet8M": "bezzam/diffusercam-mirflickr-unrolled-admm5-unet8M",
            "Unet8M+U5": "bezzam/diffusercam-mirflickr-unet8M-unrolled-admm5",
            "Unet2M+MMCN+Unet2M": "bezzam/diffusercam-mirflickr-unet2M-mmcn-unet2M",
            "Unet4M+U20+Unet4M": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm20-unet4M",
            "Unet4M+U10+Unet4M": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm10-unet4M",
            "Unet4M+U5+Unet4M_psfNN": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psfNN",
            # training with PSF noise
            "U5+Unet8M_psf0dB": "bezzam/diffusercam-mirflickr-unrolled-admm5-unet8M-psf0dB",
            "U5+Unet8M_psf-5dB": "bezzam/diffusercam-mirflickr-unrolled-admm5-unet8M-psf-5dB",
            "U5+Unet8M_psf-10dB": "bezzam/diffusercam-mirflickr-unrolled-admm5-unet8M-psf-10dB",
            "U5+Unet8M_psf-20dB": "bezzam/diffusercam-mirflickr-unrolled-admm5-unet8M-psf-20dB",
            "Unet4M+U5+Unet4M_psf-0dB": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psf-0dB",
            "Unet4M+U5+Unet4M_psf-5dB": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psf-5dB",
            "Unet4M+U5+Unet4M_psf-10dB": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psf-10dB",
            "Unet4M+U5+Unet4M_psf-20dB": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psf-20dB",
            "Unet4M+U5+Unet4M_psfNN_psf-0dB": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psfNN-psf-0dB",
            "Unet4M+U5+Unet4M_psfNN_psf-10dB": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psfNN-psf-10dB",
            "Unet4M+U5+Unet4M_psfNN_psf-20dB": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psfNN-psf-20dB",
            # training with noise
            "U5+Unet8M_10db": "bezzam/diffusercam-mirflickr-unrolled-admm5-unet8M-10db",
            "U5+Unet8M_40db": "bezzam/diffusercam-mirflickr-unrolled-admm5-unet8M-40db",
            "Unet4M+U5+Unet4M_10db": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-10db",
            "Unet4M+U5+Unet4M_40db": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-40db",
            # fine-tuning tapecam
            "Unet4M+U5+Unet4M_ft_tapecam": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-ft-tapecam",
            "Unet4M+U5+Unet4M_ft_tapecam_post": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-ft-tapecam-post",
            "Unet4M+U5+Unet4M_ft_tapecam_pre": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-ft-tapecam-pre",
            # comparing with transformers, with ADAMW optimizer (rest with ADAM)
            "Unet4M+U5+Unet4M_adamw": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-adamw",
            "Unet4M+U5+Unet4M_psfNN_adamw": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psfNN-adamw",
            "U5+Transformer8M": "bezzam/diffusercam-mirflickr-unrolled-admm5-transformer8M",
            "Transformer4M+U5+Transformer4M": "bezzam/diffusercam-mirflickr-transformer4M-unrolled-admm5-transformer4M",
            "Transformer4M+U5+Transformer4M_psfNN": "bezzam/difusercam-mirflickr-transformer4M-unrolled-admm5-transformer4M-psfNN",
            # (~11.6M param) comparing with SVDeconvNet, with ADAMW optimizer (full resolution images)
            "Unet6M+U5+Unet6M_fullres": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-fullres",
            "Unet6M+U5+Unet6M_psfNN_fullres": "bezzam/diffusercam-mirflickr-unet4M-unrolled-admm5-unet4M-psfNN-fullres",
            "SVDecon+UNet8M": "bezzam/diffusercam-mirflickr-svdecon-unet4M",
            "Unet4M+SVDecon+Unet4M": "bezzam/diffusercam-mirflickr-unet4M-svdecon-unet4M",
        },
        "mirflickr_sim": {
            "Unet4M+U5+Unet4M": "bezzam/diffusercam-mirflickr-sim-unet4M-unrolled-admm5-unet4M",
            "Unet4M+U5+Unet4M_ft_tapecam": "bezzam/diffusercam-mirflickr-sim-unet4M-unrolled-admm5-unet4M-ft-tapecam",
            "Unet4M+U5+Unet4M_ft_tapecam_post": "bezzam/diffusercam-mirflickr-sim-unet4M-unrolled-admm5-unet4M-ft-tapecam-post",
            "Unet4M+U5+Unet4M_ft_tapecam_pre": "bezzam/diffusercam-mirflickr-sim-unet4M-unrolled-admm5-unet4M-ft-tapecam-pre",
            "Unet4M+U5+Unet4M_ft_digicam_multi_post": "bezzam/diffusercam-mirflickr-sim-unet4M-unrolled-admm5-unet4M-ft-digicam-multi-post",
            "Unet4M+U5+Unet4M_ft_digicam_multi_pre": "bezzam/diffusercam-mirflickr-sim-unet4M-unrolled-admm5-unet4M-ft-digicam-multi-pre",
            "Unet4M+U5+Unet4M_ft_digicam_multi": "bezzam/diffusercam-mirflickr-sim-unet4M-unrolled-admm5-unet4M-ft-digicam-multi",
        },
    },
    "digicam": {
        "celeba_26k": {
            "unrolled_admm10": "bezzam/digicam-celeba-unrolled-admm10",
            "unrolled_admm10_ft_psf": "bezzam/digicam-celeba-unrolled-admm10-ft-psf",
            "unet8M": "bezzam/digicam-celeba-unet8M",
            "TrainInv+Unet8M": "bezzam/digicam-celeba-trainable-inv-unet8M",
            "unrolled_admm10_post8M": "bezzam/digicam-celeba-unrolled-admm10-post8M",
            "unrolled_admm10_ft_psf_post8M": "bezzam/digicam-celeba-unrolled-admm10-ft-psf-post8M",
            "pre8M_unrolled_admm10": "bezzam/digicam-celeba-pre8M-unrolled-admm10",
            "pre4M_unrolled_admm10_post4M": "bezzam/digicam-celeba-pre4M-unrolled-admm10-post4M",
            "pre4M_unrolled_admm10_ft_psf_post4M": "bezzam/digicam-celeba-pre4M-unrolled-admm10-ft-psf-post4M",
            "Unet4M+TrainInv+Unet4M": "bezzam/digicam-celeba-unet4M-trainable-inv-unet4M",
            # baseline benchmarks which don't have model file but use ADMM
            "admm_measured_psf": "bezzam/digicam-celeba-admm-measured-psf",
            "admm_simulated_psf": "bezzam/digicam-celeba-admm-simulated-psf",
            # TCI submission (using waveprop simulation)
            "U5+Unet8M_wave": "bezzam/digicam-celeba-unrolled-admm5-unet8M",
            "Unet8M+U5_wave": "bezzam/digicam-celeba-unet8M-unrolled-admm5",
            "TrainInv+Unet8M_wave": "bezzam/digicam-celeba-trainable-inv-unet8M_wave",
            "MWDN8M_wave": "bezzam/digicam-celeba-mwnn-8M",
            "MMCN4M+Unet4M_wave": "bezzam/digicam-celeba-mmcn-unet4M",
            "Unet2M+MWDN6M_wave": "bezzam/digicam-celeba-unet2M-mwdn-6M",
            "Unet4M+TrainInv+Unet4M_wave": "bezzam/digicam-celeba-unet4M-trainable-inv-unet4M_wave",
            "Unet2M+MMCN+Unet2M_wave": "bezzam/digicam-celeba-unet2M-mmcn-unet2M",
            "Unet4M+U5+Unet4M_wave": "bezzam/digicam-celeba-unet4M-unrolled-admm5-unet4M",
            "Unet4M+U10+Unet4M_wave": "bezzam/digicam-celeba-unet4M-unrolled-admm10-unet4M",
            "Unet4M+U5+Unet4M_wave_psfNN": "bezzam/digicam-celeba-unet4M-unrolled-admm5-unet4M-wave-psfNN",
        },
        "mirflickr_single_25k": {
            # simulated PSF (without waveprop, with deadspace)
            "U10": "bezzam/digicam-mirflickr-single-25k-unrolled-admm10",
            "Unet8M": "bezzam/digicam-mirflickr-single-25k-unet8M",
            "TrainInv+Unet8M": "bezzam/digicam-mirflickr-single-25k-trainable-inv-unet8M",
            "U10+Unet8M": "bezzam/digicam-mirflickr-single-25k-unrolled-admm10-unet8M",
            "Unet4M+TrainInv+Unet4M": "bezzam/digicam-mirflickr-single-25k-unet4M-trainable-inv-unet4M",
            "Unet4M+U10+Unet4M": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm10-unet4M",
            # simulated PSF (with waveprop, with deadspace)
            "U10_wave": "bezzam/digicam-mirflickr-single-25k-unrolled-admm10-wave",
            "U10+Unet8M_wave": "bezzam/digicam-mirflickr-single-25k-unrolled-admm10-unet8M-wave",
            "Unet8M_wave": "bezzam/digicam-mirflickr-single-25k-unet8M-wave",
            "Unet8M_wave_v2": "bezzam/digicam-mirflickr-single-25k-unet8M-wave-v2",
            "Unet4M+U10+Unet4M_wave": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm10-unet4M-wave",
            "TrainInv+Unet8M_wave": "bezzam/digicam-mirflickr-single-25k-trainable-inv-unet8M-wave",
            "U5+Unet8M_wave": "bezzam/digicam-mirflickr-single-25k-unrolled-admm5-unet8M-wave",
            "Unet8M+U5_wave": "bezzam/digicam-mirflickr-single-25k-unet8M-unrolled-admm5-wave",
            "Unet4M+U5+Unet4M_wave": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm5-unet4M-wave",
            "Unet4M+U5+Unet4M_wave_psfNN": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm5-unet4M-wave-psfNN",
            "MWDN8M_wave": "bezzam/digicam-mirflickr-single-25k-mwdn-8M",
            "MMCN4M+Unet4M_wave": "bezzam/digicam-mirflickr-single-25k-mmcn-unet4M",
            "Unet2M+MMCN+Unet2M_wave": "bezzam/digicam-mirflickr-single-25k-unet2M-mmcn-unet2M-wave",
            "Unet4M+TrainInv+Unet4M_wave": "bezzam/digicam-mirflickr-single-25k-unet4M-trainable-inv-unet4M-wave",
            "Unet2M+MWDN6M_wave": "bezzam/digicam-mirflickr-single-25k-unet2M-mwdn-6M",
            "Unet4M+U5+Unet4M_wave_aux1": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm5-unet4M-wave-aux1",
            "Unet4M+U5+Unet4M_wave_flips": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm5-unet4M-wave-flips",
            "Unet4M+U5+Unet4M_wave_flips_rotate10": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm5-unet4M-wave-flips-rotate10",
            # measured PSF
            "Unet4M+U10+Unet4M_measured": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm10-unet4M-measured",
            # simulated PSF (with waveprop, no deadspace)
            "Unet4M+U10+Unet4M_wave_nodead": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm10-unet4M-wave-nodead",
            # simulated PSF (without waveprop, no deadspace)
            "Unet4M+U10+Unet4M_nodead": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm10-unet4M-nodead",
            # finetune
            "Unet4M+U5+Unet4M_ft_flips": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm5-unet4M-ft-flips",
            "Unet4M+U5+Unet4M_ft_flips_rotate10": "bezzam/digicam-mirflickr-single-25k-unet4M-unrolled-admm5-unet4M-ft-flips-rotate10",
        },
        "mirflickr_multi_25k": {
            # simulated PSFs (without waveprop, with deadspace)
            "Unet8M": "bezzam/digicam-mirflickr-multi-25k-unet8M",
            "Unet8M_wave_v2": "bezzam/digicam-mirflickr-multi-25k-unet8M-wave-v2",
            "Unet4M+U10+Unet4M": "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm10-unet4M",
            # simulated PSF (with waveprop, with deadspace)
            "Unet4M+U10+Unet4M_wave": "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm10-unet4M-wave",
            "Unet4M+U5+Unet4M_wave": "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm5-unet4M-wave",
            "Unet4M+U5+Unet4M_wave_psfNN": "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm5-unet4M-wave-psfNN",
            "Unet4M+U5+Unet4M_wave_aux1": "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm5-unet4M-wave-aux1",
            "Unet4M+U5+Unet4M_wave_flips": "bezzam/digicam-mirflickr-multi-25k-unet4M-unrolled-admm5-unet4M-wave-flips",
        },
    },
    "tapecam": {
        "mirflickr": {
            "U5+Unet8M": "bezzam/tapecam-mirflickr-unrolled-admm5-unet8M",
            "Unet8M+U5": "bezzam/tapecam-mirflickr-unet8M-unrolled-admm5",
            "TrainInv+Unet8M": "bezzam/tapecam-mirflickr-trainable-inv-unet8M",
            "MMCN4M+Unet4M": "bezzam/tapecam-mirflickr-mmcn-unet4M",
            "MWDN8M": "bezzam/tapecam-mirflickr-mwdn-8M",
            "Unet4M+TrainInv+Unet4M": "bezzam/tapecam-mirflickr-unet4M-trainable-inv-unet4M",
            "Unet4M+U5+Unet4M": "bezzam/tapecam-mirflickr-unet4M-unrolled-admm5-unet4M",
            "Unet2M+MMCN+Unet2M": "bezzam/tapecam-mirflickr-unet2M-mmcn-unet2M",
            "Unet2M+MWDN6M": "bezzam/tapecam-mirflickr-unet2M-mwdn-6M",
            "Unet4M+U10+Unet4M": "bezzam/tapecam-mirflickr-unet4M-unrolled-admm10-unet4M",
            "Unet4M+U5+Unet4M_flips": "bezzam/tapecam-mirflickr-unet4M-unrolled-admm5-unet4M-flips",
            "Unet4M+U5+Unet4M_flips_rotate10": "bezzam/tapecam-mirflickr-unet4M-unrolled-admm5-unet4M-flips-rotate10",
            "Unet4M+U5+Unet4M_aux1": "bezzam/tapecam-mirflickr-unet4M-unrolled-admm5-unet4M-aux1",
            "Unet4M+U5+Unet4M_psfNN": "bezzam/tapecam-mirflickr-unet4M-unrolled-admm5-unet4M-psfNN",
        },
    },
    "multilens": {
        "mirflickr_ambient": {
            "U5+Unet8M": "lensless/multilens-mirflickr-ambient-unrolled-admm5-unet8M",
            "U5+Unet8M_direct_sub": "lensless/multilens-mirflickr-ambient-unrolled-admm5-unet8M-direct-sub",
            "U5+Unet8M_learned_sub": "lensless/multilens-mirflickr-ambient-unrolled-admm5-unet8M-learned-sub",
            "Unet4M+U5+Unet4M": "lensless/multilens-mirflickr-ambient-unet4M-unrolled-admm5-unet4M",
            "Unet4M+U5+Unet4M_direct_sub": "lensless/multilens-mirflickr-ambient-unet4M-unrolled-admm5-unet4M-direct-sub",
            "Unet4M+U5+Unet4M_learned_sub": "lensless/multilens-mirflickr-ambient-unet4M-unrolled-admm5-unet4M-learned-sub",
            "Unet4M+U5+Unet4M_concat": "lensless/multilens-mirflickr-ambient-unet4M-unrolled-admm5-unet4M-concat-ext",
            "Unet4M+U5+Unet4M_concat_psfNN": "lensless/multilens-mirflickr-ambient-unet4M-unrolled-admm5-unet4M-concat-psfNN",
            "TrainInv+Unet8M": "lensless/multilens-mirflickr-ambient-trainable-inv-unet8M",
            "TrainInv+Unet8M_learned_sub": "lensless/multilens-mirflickr-ambient-trainable-inv-unet8M-learned-sub",
            "Unet4M+TrainInv+Unet4M": "lensless/multilens-mirflickr-ambient-unet4M-trainable-inv-unet4M",
            "Unet4M+TrainInv+Unet4M_learned_sub": "lensless/multilens-mirflickr-ambient-unet4M-trainable-inv-unet4M-learned-sub",
            "Unet4M+TrainInv+Unet4M_concat": "lensless/multilens-mirflickr-ambient-unet4M-trainable-inv-unet4M-concat-ext",
            "TrainInv+Unet8M_direct_sub": "lensless/multilens-mirflickr-ambient-trainable-inv-unet8M-direct-sub",
            "Unet4M+TrainInv+Unet4M_direct_sub": "lensless/multilens-mirflickr-ambient-unet4M-trainable-inv-unet4M-direct-sub",
        }
    },
}


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        # remove `module.` from k
        if "module." in k:
            name = k.replace("module.", "")
            # name = k[7:] # remove `module.`
            new_state_dict[name] = v

    return new_state_dict


def download_model(camera, dataset, model, local_model_dir=None):

    """
    Download model from model_dict (if needed).

    Parameters
    ----------
    dataset : str
        Dataset used for training.
    model_name : str
        Name of model.
    """

    if local_model_dir is None:
        local_model_dir = model_dir_path

    if camera not in model_dict:
        raise ValueError(f"Camera {camera} not found in model_dict.")

    if dataset not in model_dict[camera]:
        raise ValueError(f"Dataset {dataset} not found in model_dict.")

    if model not in model_dict[camera][dataset]:
        raise ValueError(f"Model {model} not found in model_dict.")

    repo_id = model_dict[camera][dataset][model]
    model_dir = os.path.join(local_model_dir, camera, dataset, model)

    if not os.path.exists(model_dir):
        snapshot_download(repo_id=repo_id, local_dir=model_dir)

    return model_dir


def load_model(
    model_path,
    psf,
    device="cpu",
    device_ids=None,
    legacy_denoiser=False,
    verbose=True,
    skip_pre=False,
    skip_post=False,
    train_last_layer=False,
    return_intermediate=False,
):

    """
    Load best model from model path.

    Parameters
    ----------
    model_path : str
        Path to model.
    psf : py:class:`~torch.Tensor`
        PSF tensor.
    device : str
        Device to load model on.
    """

    # load Hydra config
    config_path = os.path.join(model_path, ".hydra", "config.yaml")
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # load learning mask
    downsample = config["files"]["downsample"] * 4  # TODO: particular to DiffuserCam?
    mask = prep_trainable_mask(config, psf, downsample=downsample)
    if mask is not None:

        if config["trainable_mask"]["mask_type"] == "TrainablePSF":
            # print("loading best PSF...")

            psf_learned = np.load(os.path.join(model_path, "psf_epochBEST.npy"))
            psf_learned = torch.Tensor(psf_learned).to(psf).unsqueeze(0)

            # -- set values and get new PSF
            with torch.no_grad():
                mask._mask = torch.nn.Parameter(torch.tensor(psf_learned))
                psf = mask.get_psf().to(device)

    # load best model config
    model_checkpoint = os.path.join(model_path, "recon_epochBEST")
    assert os.path.exists(model_checkpoint), "Checkpoint does not exist"
    if verbose:
        print("Loading checkpoint from : ", model_checkpoint)
    model_state_dict = torch.load(model_checkpoint, map_location=device)

    if config["files"].get("psf_snr", None) is not None:
        # overwrite PSF with noisy PSF used during training
        psf_path = os.path.join(model_path, "psf.pt")
        assert os.path.exists(psf_path), "PSF does not exist"
        psf = torch.load(psf_path, map_location=device)

    # load model
    pre_process = None
    post_process = None

    background_network = None
    if config["reconstruction"].get("direct_background_subtraction", False):
        assert config["reconstruction"].get("learned_background_subtraction", None) is False
    elif config["reconstruction"].get("learned_background_subtraction", False):
        assert config["reconstruction"].get("direct_background_subtraction", None) is False

        # create UnetRes for background subtraction
        background_network, _ = create_process_network(
            network="UnetRes",
            depth=len(config["reconstruction"]["learned_background_subtraction"]),
            nc=config["reconstruction"]["learned_background_subtraction"],
            device=device,
            device_ids=device_ids,
        )
    elif config["reconstruction"].get("integrated_background_subtraction", False):
        assert config["reconstruction"]["direct_background_subtraction"] is False
        assert config["reconstruction"]["learned_background_subtraction"] is False
        assert pre_process is None

        background_network = IntegratedBackgroundSub(
            in_channels=3,
            out_channels=3,
            input_shape=psf.shape,
            nc=config["reconstruction"]["integrated_background_subtraction"],
            down_subtraction=config["reconstruction"]["down_subtraction"],
        )

    # network for PSF
    psf_network = None
    if config["reconstruction"].get("psf_network", None):
        # create UnetRes for PSF
        psf_network, _ = create_process_network(
            network="UnetRes",
            depth=len(config["reconstruction"]["psf_network"]),
            nc=config["reconstruction"]["psf_network"],
            device=device,
            device_ids=device_ids,
        )

    if config["reconstruction"].get("init", None):

        init_model = config["reconstruction"]["init"]
        assert config["reconstruction"].get("init_processors", None) is None

        param = init_model.split(":")
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
            train_last_layer=config["reconstruction"]["post_process"]["train_last_layer"],
        )

    else:

        if config["reconstruction"]["pre_process"]["network"] is not None:

            pre_process, _ = create_process_network(
                network=config["reconstruction"]["pre_process"]["network"],
                device=device,
                input_background=config["reconstruction"].get("unetres_input_background", False),
                # unetres param
                depth=config["reconstruction"]["pre_process"]["depth"],
                nc=config["reconstruction"]["pre_process"]["nc"]
                if "nc" in config["reconstruction"]["pre_process"].keys()
                else None,
                # restormer parameters
                restormer_params=config["reconstruction"]["pre_process"].get(
                    "restormer_params", None
                ),
            )

        if config["reconstruction"]["post_process"]["network"] is not None:

            post_process, _ = create_process_network(
                network=config["reconstruction"]["post_process"]["network"],
                device=device,
                # get from dict
                concatenate_compensation=config["reconstruction"]["compensation"][-1]
                if config["reconstruction"].get("compensation", None) is not None
                else False,
                # unetres param
                depth=config["reconstruction"]["post_process"]["depth"],
                nc=config["reconstruction"]["post_process"]["nc"]
                if "nc" in config["reconstruction"]["post_process"].keys()
                else None,
                # restormer parameters
                restormer_params=config["reconstruction"]["post_process"].get(
                    "restormer_params", None
                ),
            )

            if train_last_layer:
                for param in post_process.parameters():
                    for name, param in post_process.named_parameters():
                        if "m_tail" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

        if config["reconstruction"]["method"] == "unrolled_admm":
            recon = UnrolledADMM(
                psf if mask is None else psf_learned,
                pre_process=pre_process,
                post_process=post_process,
                n_iter=config["reconstruction"]["unrolled_admm"]["n_iter"],
                skip_unrolled=config["reconstruction"]["skip_unrolled"],
                legacy_denoiser=legacy_denoiser,
                skip_pre=skip_pre,
                skip_post=skip_post,
                compensation=config["reconstruction"].get("compensation", None),
                psf_network=psf_network,
                psf_residual=config["reconstruction"].get("psf_residual", False),
                compensation_residual=config["reconstruction"].get("compensation_residual", False),
                direct_background_subtraction=config["reconstruction"].get(
                    "direct_background_subtraction", False
                ),
                integrated_background_subtraction=config["reconstruction"].get(
                    "integrated_background_subtraction", False
                ),
                background_network=background_network,
                return_intermediate=return_intermediate,
            )
        elif config["reconstruction"]["method"] == "trainable_inv":
            recon = TrainableInversion(
                psf,
                pre_process=pre_process,
                post_process=post_process,
                K=config["reconstruction"]["trainable_inv"]["K"],
                legacy_denoiser=legacy_denoiser,
                skip_pre=skip_pre,
                skip_post=skip_post,
                direct_background_subtraction=config["reconstruction"].get(
                    "direct_background_subtraction", False
                ),
                integrated_background_subtraction=config["reconstruction"].get(
                    "integrated_background_subtraction", False
                ),
                background_network=background_network,
                return_intermediate=return_intermediate,
            )
        elif config["reconstruction"]["method"] == "multi_wiener":

            if config["files"].get("single_channel_psf", False):

                if torch.sum(psf[..., 0] - psf[..., 1]) != 0:
                    # need to sum difference channels
                    raise ValueError("PSF channels are not the same")
                    # psf = np.sum(psf, axis=3)

                else:
                    psf = psf[..., 0].unsqueeze(-1)
                psf_channels = 1
            else:
                psf_channels = 3

            recon = MultiWiener(
                in_channels=3,
                out_channels=3,
                psf=psf,
                psf_channels=psf_channels,
                nc=config["reconstruction"]["multi_wiener"]["nc"],
                pre_process=pre_process,
            )
            recon.to(device)

        elif config["reconstruction"]["method"] == "svdeconvnet":
            psf_learned = psf_learned[0]  # remove singleton dimension
            recon = SVDeconvNet(
                psf[0].unsqueeze(
                    0
                ),  # set one of PSF (just for shape) but will be overwritten later
                K=config["reconstruction"]["svdeconvnet"]["K"],
                pre_process=pre_process,
                post_process=post_process,
                background_network=background_network,
                return_intermediate=return_intermediate,
                direct_background_subtraction=config["reconstruction"].get(
                    "direct_background_subtraction", False
                ),
                integrated_background_subtraction=config["reconstruction"].get(
                    "integrated_background_subtraction", False
                ),
            )

    if mask is not None:
        psf_learned = torch.nn.Parameter(psf_learned)
        recon._set_psf(psf_learned)

    if config.get("device_ids", None) is not None or isinstance(config.get("torch_device"), list):
        model_state_dict = remove_data_parallel(model_state_dict)

    # hotfixes for loading models
    if config["reconstruction"]["method"] == "multi_wiener":
        # replace "avgpool_conv" with "pool_conv"
        model_state_dict = {
            k.replace("avgpool_conv", "pool_conv"): v for k, v in model_state_dict.items()
        }

    recon.load_state_dict(model_state_dict)

    if device_ids is not None:
        if recon.pre_process is not None and background_network is None:
            pre_proc = torch.nn.DataParallel(recon.pre_process_model, device_ids=device_ids)
            pre_proc = pre_proc.to(device)
            recon.set_pre_process(pre_proc)
        if recon.post_process is not None:
            post_proc = torch.nn.DataParallel(recon.post_process_model, device_ids=device_ids)
            post_proc = post_proc.to(device)
            recon.set_post_process(post_proc)
        recon = MyDataParallel(recon, device_ids=device_ids)
    recon.to(device)
    return recon
