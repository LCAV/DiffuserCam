import os
import hydra
from hydra.utils import to_absolute_path
import subprocess
import numpy as np
import time
from pprint import pprint
from lensless.utils.plot import plot_image, pixel_histogram
from lensless.utils.io import save_image
from lensless.utils.image import resize, print_image_info
import cv2
import matplotlib.pyplot as plt
from lensless import FISTA, ADMM
from lensless.hardware.utils import check_username_hostname, display
from lensless.utils.io import load_image, load_psf
import omegaconf
from lensless.hardware.slm import set_programmable_mask, adafruit_sub2full
from lensless.hardware.trainable_mask import AdafruitLCD
from torch import from_numpy


@hydra.main(version_base=None, config_path="../configs", config_name="demo")
def demo(config):

    check_username_hostname(config.rpi.username, config.rpi.hostname)
    RPI_USERNAME = config.rpi.username
    RPI_HOSTNAME = config.rpi.hostname

    display_fp = to_absolute_path(config.fp)
    if config.save:
        if config.output is not None:
            # make sure output directory exists
            os.makedirs(config.output, exist_ok=True)
            save = config.output
        else:
            save = os.getcwd()
    else:
        save = False

    # 1) Copy file to Raspberry Pi
    print("\nCopying over picture...")
    display(fp=display_fp, rpi_username=RPI_USERNAME, rpi_hostname=RPI_HOSTNAME, **config.display)

    # 2) (If DigiCam) set mask pattern
    mask = None
    flipud = False
    if isinstance(config.camera.psf, omegaconf.dictconfig.DictConfig):
        print("\nSetting mask pattern...")
        np.random.seed(config.camera.psf.seed % (2**32 - 1))
        mask_vals = np.random.uniform(0, 1, config.camera.psf.mask_shape)
        full_pattern = adafruit_sub2full(
            mask_vals,
            center=config.camera.psf.mask_center,
        )
        set_programmable_mask(
            full_pattern,
            device=config.camera.psf.device,
            rpi_username=RPI_USERNAME,
            rpi_hostname=RPI_HOSTNAME,
        )
        mask_vals_torch = from_numpy(mask_vals.astype(np.float32))
        flipud = config.camera.psf.flipud
        mask = AdafruitLCD(
            initial_vals=mask_vals_torch,
            sensor=config.capture.sensor,
            slm=config.camera.psf.device,
            downsample=config.recon.downsample,
            flipud=flipud,
        )

    # 3) Take picture
    time.sleep(config.capture.delay)  # for picture to display

    remote_fn = "remote_capture"
    print("\nTaking picture...")
    pic_command = (
        f"{config.rpi.python} {config.capture.script} sensor={config.capture.sensor} bayer={config.capture.bayer} fn={remote_fn} exp={config.capture.exp} iso={config.capture.iso} "
        f"config_pause={config.capture.config_pause} sensor_mode={config.capture.sensor_mode} nbits_out={config.capture.nbits_out} "
        f"legacy={config.capture.legacy} rgb={config.capture.rgb} gray={config.capture.gray} "
    )
    if config.capture.nbits > 8:
        pic_command += " sixteen=True"
    if config.capture.down:
        pic_command += f" down={config.capture.down}"
    if config.capture.awb_gains:
        pic_command += f" awb_gains=[{config.capture.awb_gains[0]},{config.capture.awb_gains[1]}]"

    print(f"COMMAND : {pic_command}")
    ssh = subprocess.Popen(
        ["ssh", "%s@%s" % (RPI_USERNAME, RPI_HOSTNAME), pic_command],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = ssh.stdout.readlines()
    error = ssh.stderr.readlines()

    if (
        error != [] and config.capture.legacy
    ):  # new camera software seems to return error even if it works
        print("ERROR: %s" % error)
        return
    if result == []:
        error = ssh.stderr.readlines()
        print("ERROR: %s" % error)
        return
    else:
        result = [res.decode("UTF-8") for res in result]
        result = [res for res in result if len(res) > 3]
        result_dict = dict()
        for res in result:
            _key = res.split(":")[0].strip()
            _val = "".join(res.split(":")[1:]).strip()
            result_dict[_key] = _val
        # result_dict = dict(map(lambda s: map(str.strip, s.split(":")), result))
        print("COMMAND OUTPUT : ")
        pprint(result_dict)

    if (
        "RPi distribution" in result_dict.keys()
        and "bullseye" in result_dict["RPi distribution"]
        and not config.capture.legacy
    ):
        assert (
            not config.capture.rgb or not config.capture.gray
        ), "RGB and gray not supported for RPi HQ sensor"

        if config.capture.bayer:

            assert config.capture.down is None

            # copy over DNG file
            remotefile = f"~/{remote_fn}.dng"
            localfile = os.path.join(save, f"{config.capture.raw_data_fn}.dng")
            print(f"\nCopying over picture as {localfile}...")
            os.system('scp "%s@%s:%s" %s' % (RPI_USERNAME, RPI_HOSTNAME, remotefile, localfile))

            img = load_image(
                localfile,
                verbose=True,
                bayer=config.capture.bayer,
                nbits_out=config.capture.nbits_out,
            )

            # print image properties
            print_image_info(img)

            # save as PNG
            png_out = f"{config.capture.raw_data_fn}.png"
            print(f"Saving RGB file as: {png_out}")
            cv2.imwrite(png_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        else:

            remotefile = f"~/{remote_fn}.png"
            localfile = f"{config.capture.raw_data_fn}.png"
            if save:
                localfile = os.path.join(save, localfile)
            print(f"\nCopying over picture as {localfile}...")
            os.system('scp "%s@%s:%s" %s' % (RPI_USERNAME, RPI_HOSTNAME, remotefile, localfile))

            img = load_image(localfile, verbose=True)

    # legacy software running on RPi
    else:
        # copy over file
        # more pythonic? https://stackoverflow.com/questions/250283/how-to-scp-in-python
        remotefile = f"~/{remote_fn}.png"
        localfile = f"{config.capture.raw_data_fn}.png"
        if save:
            localfile = os.path.join(save, localfile)
        print(f"\nCopying over picture as {localfile}...")
        os.system('scp "%s@%s:%s" %s' % (RPI_USERNAME, RPI_HOSTNAME, remotefile, localfile))

        if config.capture.rgb or config.capture.gray:
            img = load_image(localfile, verbose=True)

        else:

            if config.capture.bayer:
                red_gain = config.camera.red_gain
                blue_gain = config.camera.blue_gain
            else:
                red_gain = None
                blue_gain = None

            # load image
            print("\nLoading picture...")

            img = load_image(
                localfile,
                verbose=True,
                bayer=config.capture.bayer,
                blue_gain=blue_gain,
                red_gain=red_gain,
                nbits_out=config.capture.nbits_out,
            )

            # write RGB data
            if not config.capture.bayer:
                cv2.imwrite(localfile, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # plot image and histogram (useful to check clipping)
    ax = plot_image(img, gamma=config.capture.gamma)
    ax.set_title("Raw data")
    if save:
        plt.savefig(os.path.join(save, "raw.png"))
    pixel_histogram(img)
    if save:
        plt.savefig(os.path.join(save, "histogram.png"))

    # 4) Reconstruct

    # -- prepare data
    if mask is not None:
        psf = mask.get_psf().detach().numpy()
        bg = np.zeros(psf.shape[-1])
    else:
        psf, bg = load_psf(
            to_absolute_path(config.camera.psf),
            downsample=config.recon.downsample,
            return_float=True,
            return_bg=True,
            dtype=config.recon.dtype,
        )
        psf = np.array(psf, dtype=config.recon.dtype)
    ax = plot_image(psf[0], gamma=config.recon.gamma)
    ax.set_title("PSF")
    if save:
        plt.savefig(os.path.join(save, "psf.png"))

    data = np.array(img, dtype=config.recon.dtype)
    data -= bg
    data = np.clip(data, a_min=0, a_max=data.max())

    if len(data.shape) == 3:
        data = data[np.newaxis, :, :, :]
    elif len(data.shape) == 2:
        data = data[np.newaxis, :, :, np.newaxis]

    if data.shape != psf.shape:
        # in DiffuserCam dataset, images are already reshaped
        data = resize(data, shape=psf.shape)
    data /= np.linalg.norm(data.ravel())
    data = np.array(data, dtype=config.recon.dtype)

    if config.recon.use_torch:
        import torch

        if config.recon.dtype == "float32":
            torch_dtype = torch.float32
        elif config.recon.dtype == "float64":
            torch_dtype = torch.float64
        else:
            raise ValueError("dtype must be float32 or float64")

        psf = torch.from_numpy(psf).type(torch_dtype).to(config.recon.torch_device)
        data = torch.from_numpy(data).type(torch_dtype).to(config.recon.torch_device)
        if flipud:
            data = torch.rot90(data, dims=(-3, -2), k=2)
    else:
        if flipud:
            data = np.rot90(data, k=2, axes=(-3, -2))

    # -- apply algo
    start_time = time.time()

    if config.recon.algo == "fista":
        algo_params = config.recon.fista
        recon = FISTA(
            psf,
            **algo_params,
        )
    elif config.recon.algo == "admm":
        algo_params = config.recon.admm
        recon = ADMM(
            psf,
            **algo_params,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config.recon.algo}")

    recon.set_data(data)
    res = recon.apply(
        gamma=config.recon.gamma,
        save=save,
        plot=config.plot,
        disp_iter=algo_params["disp_iter"],
    )
    print(f"Processing time : {time.time() - start_time} s")

    if config.plot:
        final_image = res[0]
    else:
        final_image = res

    # save final image ax
    if save:

        # take first depth
        final_image = final_image[0]
        if config.recon.use_torch:
            img = final_image.cpu().numpy()
        else:
            img = final_image

        if config.postproc.crop_hor is not None:
            img = img[
                :,
                int(config.postproc.crop_hor[0] * img.shape[1]) : int(
                    config.postproc.crop_hor[1] * img.shape[1]
                ),
            ]
        if config.postproc.crop_vert is not None:
            img = img[
                int(config.postproc.crop_vert[0] * img.shape[0]) : int(
                    config.postproc.crop_vert[1] * img.shape[0]
                ),
                :,
            ]

        output_fp = os.path.join(save, "reconstructed.png")
        save_image(img, output_fp)

    # clean up
    os.remove(localfile)

    if config.plot:
        plt.show()

    return save


if __name__ == "__main__":
    demo()
