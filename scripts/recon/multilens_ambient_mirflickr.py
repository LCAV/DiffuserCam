import hydra
from hydra.utils import to_absolute_path, get_original_cwd
import yaml
import torch
from lensless import ADMM
from lensless.utils.plot import plot_image
from lensless.utils.dataset import HFDataset
import os
from lensless.utils.io import save_image, load_image
from lensless.utils.image import resize
import time
from lensless.recon.model_dict import download_model, load_model
from huggingface_hub import hf_hub_download
from lensless.utils.io import load_psf


@hydra.main(
    version_base=None, config_path="../../configs", config_name="recon_multilens_ambient_mirflickr"
)
def apply_pretrained(config):
    save = config.save
    device = config.device
    n_trials = config.n_trials
    model_name = config.model

    # load config
    if model_name == "admm":
        # take config from a trained model for dataset
        model_path = download_model(
            camera="multilens", dataset="mirflickr_ambient", model="U5+Unet8M"
        )
        config_path = os.path.join(model_path, ".hydra", "config.yaml")
        with open(config_path, "r") as stream:
            model_config = yaml.safe_load(stream)

    else:
        model_path = download_model(
            camera="multilens", dataset="mirflickr_ambient", model=model_name
        )
        config_path = os.path.join(model_path, ".hydra", "config.yaml")
        with open(config_path, "r") as stream:
            model_config = yaml.safe_load(stream)

    # load data
    if config.fn is not None:
        # check if the file exists locally
        raw_data_fp = os.path.join(get_original_cwd(), config.fn)
        if not os.path.exists(raw_data_fp):
            # load raw data
            print(f"Downloading raw data from {config.fn}")
            raw_data_fp = hf_hub_download(
                repo_id=model_config["files"]["dataset"], filename=config.fn, repo_type="dataset"
            )

        lensless = load_image(
            fp=raw_data_fp,
            return_float=True,
            as_4d=True,
            normalize=False,
        )
        if lensless.shape != psf.shape:
            lensless = resize(lensless, shape=psf.shape)
        lensless = torch.from_numpy(lensless).to(psf)

        if config.background_sub:
            # check if the background exists locally
            background_fp = os.path.join(get_original_cwd(), config.background_fn)
            if not os.path.exists(background_fp):
                # load background
                print(f"Downloading background from {config.background_fn}")
                background_fp = hf_hub_download(
                    repo_id=model_config["files"]["dataset"],
                    filename=config.background_fn,
                    repo_type="dataset",
                )

            background = load_image(
                fp=background_fp,
                return_float=True,
                as_4d=True,
                normalize=False,
            )
            if background.shape != psf.shape:
                background = resize(background, shape=psf.shape)
            background = torch.from_numpy(background).to(psf)
        else:
            # create all-zeros background
            background = torch.zeros_like(lensless)

        if config.rotate:
            lensless = torch.rot90(lensless, dims=(-3, -2), k=2)
            background = torch.rot90(background, dims=(-3, -2), k=2)

        lensed = None

    else:

        test_set = HFDataset(
            huggingface_repo=model_config["files"]["dataset"],
            psf=(
                model_config["files"]["huggingface_psf"]
                if "huggingface_psf" in model_config["files"]
                else None
            ),
            split="test",
            display_res=model_config["files"]["image_res"],
            rotate=model_config["files"]["rotate"],
            downsample=model_config["files"]["downsample"],
            alignment=model_config["alignment"],
            simulation_config=model_config["simulation"],
            force_rgb=model_config["files"].get("force_rgb", False),
            cache_dir=config.cache_dir,
        )

        lensless, lensed, background = test_set[config.idx]
        lensless = lensless.to(device)
        background = background.to(device)

    # normalize lensless and background with same factor
    max_val = lensless.max()  # TODO use 4095?
    lensless = lensless / max_val
    background = background / max_val

    # load PSF
    if config.psf is not None:
        # load PSF from file
        psf, _ = load_psf(
            os.path.join(get_original_cwd(), config.psf),
            shape=lensless.shape,
            return_float=True,
            return_bg=True,
            flip=model_config["files"]["rotate"],
            flip_ud=False,
            bg_pix=(0, 15),
            force_rgb=model_config["files"].get("force_rgb", False),
        )
        psf = torch.from_numpy(psf).to(device)
        print("PSF shape: ", psf.shape)
    else:
        test_set = HFDataset(
            huggingface_repo=model_config["files"]["dataset"],
            psf=(
                model_config["files"]["huggingface_psf"]
                if "huggingface_psf" in model_config["files"]
                else None
            ),
            split="test",
            display_res=model_config["files"]["image_res"],
            rotate=model_config["files"]["rotate"],
            downsample=model_config["files"]["downsample"],
            alignment=model_config["alignment"],
            simulation_config=model_config["simulation"],
            force_rgb=model_config["files"].get("force_rgb", False),
            cache_dir=config.cache_dir,
        )
        psf = test_set.psf.to(device)
        print("PSF shape: ", psf.shape)

    # load model
    if model_name == "admm":
        recon = ADMM(psf, n_iter=config.n_iter)
    else:
        # load best model
        recon = load_model(model_path, psf, device)

    # print data shape
    print(f"Data shape :  {lensless.shape}")

    # apply reconstruction
    start_time = time.time()
    for _ in range(n_trials):
        with torch.no_grad():

            recon.set_data(lensless)
            res = recon.apply(
                disp_iter=-1,
                save=False,
                gamma=None,
                plot=False,
                background=background if config.background_sub else None,
            )
    end_time = time.time()
    avg_time_ms = (end_time - start_time) / n_trials * 1000
    print(f"Avg inference [ms] : {avg_time_ms} ms")

    img = res[0].cpu().numpy().squeeze()

    plot_image(img)
    if lensed is not None:
        plot_image(lensed)

    if save:
        print(f"Saving images to {os.getcwd()}")

        if config.fn is not None:
            dim = config.alignment.dim
            top_left = config.alignment.top_left
            res_np = img[top_left[0] : top_left[0] + dim[0], top_left[1] : top_left[1] + dim[1]]
            idx = os.path.basename(config.fn).split(".")[0]
            if not config.background_sub:
                idx = f"{idx}_nobg"
            save_image(res_np, f"{model_name}_{idx}.png")
        else:
            idx = config.idx
            if config.crop:
                alignment = test_set.alignment
                top_left = alignment["top_left"]
                height = alignment["height"]
                width = alignment["width"]
                res_np = img[top_left[0] : top_left[0] + height, top_left[1] : top_left[1] + width]
            else:
                res_np = img
            save_image(res_np, f"{model_name}_idx{idx}.png")
        if lensed is not None:
            lensed_np = lensed[0].cpu().numpy()
            save_image(lensed_np, f"original_idx{idx}.png")
        save_image(lensless[0].cpu().numpy(), f"lensless_{idx}.png", normalize=False)
        save_image(psf.squeeze().cpu().numpy(), f"psf.png")
        save_image(background[0].cpu().numpy(), f"background_{idx}.png", normalize=False)


if __name__ == "__main__":
    apply_pretrained()
