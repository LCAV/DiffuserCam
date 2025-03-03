"""

Telegram bot to interface with lensless camera setup.

"""

import hydra
import logging
import numpy as np
import os
from PIL import Image, ImageFont
import shutil
import pytz
from datetime import datetime
from lensless.hardware.utils import check_username_hostname
from lensless.hardware.slm import set_programmable_mask, adafruit_sub2full

# for displaying emojis
from emoji import EMOJI_DATA
from lensless.utils.io import load_psf
from pilmoji import Pilmoji

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup


TOKEN = None
RPI_USERNAME = None
RPI_HOSTNAME = None
RPI_LENSED_USERNAME = None
RPI_LENSED_HOSTNAME = None
CONFIG_FN = None
DEFAULT_ALGO = None
ALGO_TEXT = None
MASK_PARAM = None
TIME_OFFSET = None


OVERLAY_ALPHA = None
OVERLAY_1 = None
OVERLAY_2 = None
OVERLAY_3 = None

SETUP_FP = "docs/source/demo_setup.png"
INPUT_FP = "user_photo.jpg"
RAW_DATA_FP = "raw_data.png"
OUTPUT_FOLDER = "demo_lensless"
BUSY = False
# supported_algos = ["fista", "admm", "unrolled"]
supported_algos = ["fista", "admm"]
supported_input = ["mnist", "thumb", "face"]
FILES_CAPTURE_CONFIG = None
TIMEOUT = 1 * 60  # 10 minutes

BRIGHTNESS = 80
EXPOSURE = 0.02
LOW_LIGHT_THRESHOLD = 100
SATURATION_THRESHOLD = 0.05

PSF_FP = None
PSF_FP_GAMMA = os.path.join(OUTPUT_FOLDER, "psf_gamma.png")
BACKGROUND_FP = None

MAX_QUERIES_PER_DAY = 20
queries_count = dict()
WHITELIST_USERS = None


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def check_algo(algo):

    if algo not in supported_algos:
        return False
    else:
        return True


def get_user_folder(update):
    # name = update.message.from_user.full_name.replace(" ", "-")
    # user_subfolder = f"{update.message.from_user.id}_{name}"
    user_subfolder = f"{update.message.from_user.id}"
    return os.path.join(OUTPUT_FOLDER, user_subfolder)


def get_user_folder_from_query(query):
    # name = query.message.from_user.full_name.replace(" ", "-")
    # user_subfolder = f"{query.message.from_user.id}_{name}"
    user_subfolder = f"{query.message.from_user.id}"
    return os.path.join(OUTPUT_FOLDER, user_subfolder)


async def remove_busy_flag(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global BUSY
    BUSY = False


async def check_incoming_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global BUSY, queries_count

    # create folder for user
    user_folder = get_user_folder(update)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder, exist_ok=True)

        if MASK_PARAM is not None:

            import torch
            from lensless.hardware.slm import set_programmable_mask, adafruit_sub2full
            from lensless.hardware.trainable_mask import AdafruitLCD
            from lensless.utils.io import save_image

            user_id = int(os.path.basename(user_folder))
            np.random.seed(user_id % (2**32 - 1))  # TODO set user ID as seed
            mask_vals = np.random.uniform(0, 1, MASK_PARAM.mask_shape)

            # simulate PSF
            full_pattern = adafruit_sub2full(
                mask_vals,
                center=MASK_PARAM.mask_center,
            )
            set_programmable_mask(
                full_pattern,
                device=MASK_PARAM.device,
                rpi_username=RPI_USERNAME,
                rpi_hostname=RPI_HOSTNAME,
            )
            mask_vals_torch = torch.from_numpy(mask_vals.astype(np.float32))
            mask = AdafruitLCD(
                initial_vals=mask_vals_torch,
                sensor=MASK_PARAM.sensor,
                slm=MASK_PARAM.device,
                downsample=MASK_PARAM.downsample,
                flipud=MASK_PARAM.flipud,
            )
            psf = mask.get_psf().detach().numpy()

            # save PSF as PNG
            psf_fp = os.path.join(user_folder, "psf.png")
            save_image(psf[0], psf_fp)

            # save as NPY
            psf_npy_fp = os.path.join(user_folder, "psf.npy")
            np.save(psf_npy_fp, psf)

    if BUSY:
        return "System is busy. Please wait for the current job to finish and try again."

    # if message from a while ago, ignore
    utc = pytz.UTC
    now = utc.localize(datetime.now())
    message_time = update.message.date

    diff = (now - message_time).total_seconds()
    diff -= TIME_OFFSET

    if diff > TIMEOUT:
        return f"Timeout ({TIMEOUT} seconds) exceeded. Someone else may be using the system. Please send a new message."

    if len(update.message.photo) > 1:
        original_file_path = os.path.join(user_folder, INPUT_FP)
        photo_file = await update.message.photo[-1].get_file()
        await photo_file.download_to_drive(original_file_path)
        img = np.array(Image.open(original_file_path))

        # -- check if portrait
        if img.shape[0] < img.shape[1]:
            return "Please send a portrait photo."
            # await update.message.reply_text("Please send a portrait photo.", reply_to_message_id=update.message.message_id)
            # return
        else:
            await update.message.reply_text(
                "Got photo of resolution: " + str(img.shape),
                reply_to_message_id=update.message.message_id,
            )

    # check that not command
    elif update.message.text[0] != "/" and len(update.message.text) > 0:

        text = update.message.text

        if len(update.message.text) > 1 or text not in EMOJI_DATA:
            return "Supported text for display is only a single emoji."

    # increment queries count
    user_id = update.message.from_user.id
    if user_id not in queries_count:
        queries_count[user_id] = 1
    else:
        queries_count[user_id] += 1
        if user_id not in WHITELIST_USERS:
            if queries_count[user_id] > MAX_QUERIES_PER_DAY:
                return f"Maximum number of queries per day ({MAX_QUERIES_PER_DAY}) exceeded. Please try again tomorrow."
            return

    # print user
    print("User: ", update.message.from_user, update.message.from_user.id)
    print("Queries count: ", queries_count[user_id])

    # reset at midnight
    if now.hour == 0 and now.minute == 0:
        queries_count = dict()

    BUSY = True

    return


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    # await update.message.reply_html(
    #     ,
    # )

    await update.message.reply_photo(SETUP_FP, caption=f"Hi {user.first_name}! " + HELP_TEXT)
    await update.message.reply_text(ALGO_TEXT)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_photo(SETUP_FP, caption=HELP_TEXT)
    await update.message.reply_text(ALGO_TEXT)


async def fista(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    await reconstruct(update, context, algo="fista")


async def admm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    await reconstruct(update, context, algo="admm")


async def unrolled(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    await reconstruct(update, context, algo="unrolled")


async def unet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    await reconstruct(update, context, algo="unet")


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    1. Get photo from user
    2. Send to display
    3. Capture measurement
    4. Reconstruct
    """

    global BUSY, EXPOSURE

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    algo = update.message.caption
    if algo is not None:
        algo = algo.lower()
    else:
        algo = DEFAULT_ALGO

    if check_algo(algo):

        # # call python script for full process
        # os.system(f"python scripts/demo.py plot=False fp={INPUT_FP} output={OUTPUT_FOLDER}")

        # -- send to display
        user_folder = get_user_folder(update)
        original_file_path = os.path.join(user_folder, INPUT_FP)
        os.system(
            f"python scripts/measure/remote_display.py -cn {CONFIG_FN} fp={original_file_path} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
        )
        await update.message.reply_text(
            "Image sent to display.", reply_to_message_id=update.message.message_id
        )

        await take_picture_and_reconstruct(update, context, algo)

    else:

        await update.message.reply_text(
            f"Unsupported algorithm: {algo}. Please specify from: {supported_algos}",
            reply_to_message_id=update.message.message_id,
        )

    BUSY = False


async def take_picture(update: Update, context: ContextTypes.DEFAULT_TYPE, query=None) -> None:

    # get user subfolder
    if query is not None:
        user_subfolder = get_user_folder_from_query(query)
        await query.message.reply_text(
            f"Taking picture with exposure of {EXPOSURE} seconds...",
            reply_to_message_id=query.message.message_id,
        )
    else:
        user_subfolder = get_user_folder(update)
        await update.message.reply_text(
            f"Taking picture with exposure of {EXPOSURE} seconds...",
            reply_to_message_id=update.message.message_id,
        )

    os.system(
        f"python scripts/measure/remote_capture.py -cn {CONFIG_FN} plot=False rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME} output={user_subfolder} capture.exp={EXPOSURE}"
    )


def overlay(user_subfolder):

    if OVERLAY_1 is not None or OVERLAY_2 is not None or OVERLAY_3 is not None:

        alpha = OVERLAY_ALPHA

        reconstructed_path = os.path.join(user_subfolder, "reconstructed.png")

        img1 = Image.open(reconstructed_path)
        img1 = img1.convert("RGBA")

        for overlay_config in [OVERLAY_1, OVERLAY_2, OVERLAY_3]:
            if overlay_config is not None:
                overlay_img = Image.open(overlay_config.fp)
                overlay_img = overlay_img.convert("RGBA")
                overlay_img.putalpha(alpha)
                new_width = int(img1.width * overlay_config.scaling)
                overlay_img = overlay_img.resize(
                    (new_width, int(new_width * overlay_img.height / overlay_img.width))
                )
                img1.paste(
                    overlay_img,
                    (overlay_config.position[0], overlay_config.position[1]),
                    overlay_img,
                )

        OUTPUT_FP = os.path.join(user_subfolder, "reconstructed_overlay.png")
        img1.convert("RGB").save(OUTPUT_FP)

    else:

        OUTPUT_FP = os.path.join(user_subfolder, "reconstructed.png")

    return OUTPUT_FP


async def random_mask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Set random mask and reconstruct (with ADMM).
    """

    algo = "admm"

    # get user subfolder
    user_subfolder = get_user_folder(update)

    # get seed for random mask
    seed = int(os.path.basename(user_subfolder))
    # add random number to seed
    seed += np.random.randint(0, 1000)
    os.system(
        f"python scripts/recon/demo.py -cn {CONFIG_FN} plot=False recon.algo={algo} output={user_subfolder} camera.psf.seed={seed}"
    )

    # -- send back, with watermark if provided
    OUTPUT_FP = overlay(user_subfolder)
    await update.message.reply_photo(
        OUTPUT_FP,
        caption=f"Reconstruction ({algo})",
        reply_to_message_id=update.message.message_id,
    )

    # simulate BAD PSF
    import torch
    from lensless.hardware.slm import set_programmable_mask, adafruit_sub2full
    from lensless.hardware.trainable_mask import AdafruitLCD
    from lensless.utils.io import save_image

    np.random.seed(seed % (2**32 - 1))
    mask_vals = np.random.uniform(0, 1, MASK_PARAM.mask_shape)

    # simulate PSF
    full_pattern = adafruit_sub2full(
        mask_vals,
        center=MASK_PARAM.mask_center,
    )
    set_programmable_mask(
        full_pattern, device=MASK_PARAM.device, rpi_username=RPI_USERNAME, rpi_hostname=RPI_HOSTNAME
    )
    mask_vals_torch = torch.from_numpy(mask_vals.astype(np.float32))
    mask = AdafruitLCD(
        initial_vals=mask_vals_torch,
        sensor=MASK_PARAM.sensor,
        slm=MASK_PARAM.device,
        downsample=MASK_PARAM.downsample,
        flipud=MASK_PARAM.flipud,
    )
    psf = mask.get_psf().detach().numpy()

    # save PSF as PNG
    psf_fp = os.path.join(user_subfolder, "psf_bad.png")
    save_image(psf[0], psf_fp)
    await update.message.reply_photo(
        psf_fp,
        caption="Incorrect PSF used for reconstruction",
        reply_to_message_id=update.message.message_id,
    )

    # send back false and ground truth PSF
    psf_fp = os.path.join(user_subfolder, "psf.png")
    await update.message.reply_photo(
        psf_fp,
        caption="Correct PSF (your key)",
        reply_to_message_id=update.message.message_id,
    )


async def reconstruct(update: Update, context: ContextTypes.DEFAULT_TYPE, algo, query=None) -> None:

    supported = check_algo(algo)
    if not supported:
        await update.message.reply_text(
            f"Unsupported algorithm: {algo}. Please specify from: {supported_algos}",
            reply_to_message_id=update.message.message_id,
        )
        return

    # get user subfolder
    if query is not None:
        user_subfolder = get_user_folder_from_query(query)
        update = query  # to get the reply_to_message_id
    else:
        user_subfolder = get_user_folder(update)

    # check file exists
    raw_data = os.path.join(user_subfolder, RAW_DATA_FP)
    print(raw_data)
    if not os.path.exists(raw_data):
        await update.message.reply_text(
            "No data to reconstruct. Please take a picture first.",
            reply_to_message_id=update.message.message_id,
        )
        return

    await update.message.reply_text(
        f"Reconstructing with {algo}...", reply_to_message_id=update.message.message_id
    )
    if PSF_FP is not None:
        os.system(
            f"python scripts/recon/demo.py -cn {CONFIG_FN} plot=False recon.algo={algo} output={user_subfolder} camera.psf={PSF_FP} recon.downsample=1 camera.background={BACKGROUND_FP}"
        )
    elif MASK_PARAM is not None:
        # get seed for random mask
        seed = int(os.path.basename(user_subfolder))
        os.system(
            f"python scripts/recon/demo.py -cn {CONFIG_FN} plot=False recon.algo={algo} output={user_subfolder} camera.psf.seed={seed}"
        )
    else:
        os.system(
            f"python scripts/recon/demo.py -cn {CONFIG_FN} plot=False recon.algo={algo} output={user_subfolder}"
        )

    # -- send back, with watermark if provided
    OUTPUT_FP = overlay(user_subfolder)
    await update.message.reply_photo(
        OUTPUT_FP,
        caption=f"Reconstruction ({algo})",
        reply_to_message_id=update.message.message_id,
    )


async def take_picture_and_reconstruct(
    update: Update, context: ContextTypes.DEFAULT_TYPE, algo, query=None
) -> None:

    if query is not None:
        user_subfolder = get_user_folder_from_query(query)
        responder = query.message
    else:
        user_subfolder = get_user_folder(update)
        responder = update.message

    # (if DigiCam) set mask pattern
    if MASK_PARAM is not None:
        print("Setting mask pattern...")
        # get seed for random mask
        seed = int(os.path.basename(user_subfolder))
        np.random.seed(seed % (2**32 - 1))
        mask_vals = np.random.uniform(0, 1, MASK_PARAM.mask_shape)
        full_pattern = adafruit_sub2full(
            mask_vals,
            center=MASK_PARAM.mask_center,
        )
        # setting mask
        set_programmable_mask(
            full_pattern,
            device=MASK_PARAM.device,
            rpi_username=RPI_USERNAME,
            rpi_hostname=RPI_HOSTNAME,
        )

    await take_picture(update, context, query=query)

    # check for saturation
    OUTPUT_FP = os.path.join(user_subfolder, "raw_data_8bit.png")
    # -- load picture to check for saturation
    img = np.array(Image.open(OUTPUT_FP))
    ratio = np.sum(img == 255) / np.prod(img.shape)
    if ratio > SATURATION_THRESHOLD:

        if EXPOSURE > 0.02:
            warning_message = (
                "ERROR: saturation/clipping detected in raw measurement!"
                "\nTry reducing the /brightness of the screen, or the /exposure of the camera."
            )
        else:
            warning_message = (
                "ERROR: saturation/clipping detected in raw measurement!"
                "\nTry reducing the /brightness of the screen."
            )
        await responder.reply_photo(
            OUTPUT_FP, caption=warning_message, reply_to_message_id=responder.message_id
        )
        return

    # -- reconstruct
    await reconstruct(update, context, algo, query=query)

    # # send picture of raw measurement
    # OUTPUT_FP = os.path.join(user_subfolder, "raw_data_8bit.png")
    # # -- load picture to check for saturation
    # img = np.array(Image.open(OUTPUT_FP))
    # ratio = np.sum(img == 255) / np.prod(img.shape)
    # if ratio > 0.05:

    #     if EXPOSURE > 0.02:
    #         warning_message = ("WARNING: saturation detected in raw measurement!"
    #             "\nTry reducing the /brightness of the screen, or the /exposure of the camera.")
    #     else:
    #         warning_message = ("WARNING: saturation detected in raw measurement!"
    #             "\nTry reducing the /brightness of the screen.")

    #     await responder.reply_photo(
    #         OUTPUT_FP,
    #         caption=warning_message,
    #         reply_to_message_id=responder.message_id
    #     )

    # # -- check if low light -> send warning to increase exposure
    # if np.max(img) < LOW_LIGHT_THRESHOLD:
    #     caption = "Raw measurement.\nWARNING: low light measurement. Try increasing the /exposure of the camera."
    # else:
    caption = "Raw measurement"

    # else:
    await responder.reply_photo(
        OUTPUT_FP, caption=caption, reply_to_message_id=responder.message_id
    )

    # -- send picture of setup (lensed)
    if RPI_LENSED_HOSTNAME is not None and RPI_LENSED_USERNAME is not None:
        os.system(
            f"python scripts/measure/remote_capture.py -cn {CONFIG_FN} rpi.username={RPI_LENSED_USERNAME} rpi.hostname={RPI_LENSED_HOSTNAME} plot=False capture.bayer=False capture.down=8 output={user_subfolder} capture.raw_data_fn=lensed capture.awb_gains=null"
        )
        OUTPUT_FP = os.path.join(user_subfolder, "lensed.png")
        await responder.reply_photo(
            OUTPUT_FP, caption="Picture of setup", reply_to_message_id=responder.message_id
        )


async def file_input_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    global BUSY, EXPOSURE

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    algo = DEFAULT_ALGO

    # extract config by based on file name
    file_name = update.message.text[1:]
    if file_name not in FILES_CAPTURE_CONFIG:
        await update.message.reply_text(
            f"Unsupported file: {file_name}. Please specify from: {FILES_CAPTURE_CONFIG.keys()}",
            reply_to_message_id=update.message.message_id,
        )
        return

    brightness = FILES_CAPTURE_CONFIG[file_name]["brightness"]
    EXPOSURE = FILES_CAPTURE_CONFIG[file_name]["exposure"]
    fp = FILES_CAPTURE_CONFIG[file_name]["fp"]

    # copy image to INPUT_FP
    user_folder = get_user_folder(update)
    original_file_path = os.path.join(user_folder, INPUT_FP)
    os.system(f"cp {fp} {original_file_path}")

    # -- send to display
    os.system(
        f"python scripts/measure/remote_display.py -cn {CONFIG_FN} fp={original_file_path} display.brightness={brightness} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text(
        f"Image sent to display with brightness {brightness}.",
        reply_to_message_id=update.message.message_id,
    )

    await take_picture_and_reconstruct(update, context, algo)
    BUSY = False


async def psf_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    Measure PSF through screen
    """

    global BUSY

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    vshift = -15
    psf_size = 10

    # -- send to display
    os.system(
        f"python scripts/measure/remote_display.py -cn {CONFIG_FN} display.psf={psf_size} display.vshift={vshift} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text(
        f"PSF of {psf_size}x{psf_size} pixels set on display.",
        reply_to_message_id=update.message.message_id,
    )

    # -- measurement
    os.system(
        f"python scripts/measure/remote_capture.py -cn demo_measure_psf rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "raw_data.png")
    await update.message.reply_photo(
        OUTPUT_FP,
        caption="PSF (zoom in to see pattern)",
        reply_to_message_id=update.message.message_id,
    )

    # send back ground truth PSF
    if PSF_FP is not None:
        await update.message.reply_photo(
            PSF_FP_GAMMA,
            caption="PSF used for reconstructions",
            reply_to_message_id=update.message.message_id,
        )
    elif MASK_PARAM is not None:
        # return pre-computed PSF
        user_folder = get_user_folder(update)
        psf_fp = os.path.join(user_folder, "psf.png")
        await update.message.reply_photo(
            psf_fp,
            caption="PSF used for reconstructions",
            reply_to_message_id=update.message.message_id,
        )

    BUSY = False


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""

    global BRIGHTNESS, EXPOSURE, BUSY
    BUSY = True

    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()

    if query.data == "Cancel":
        BUSY = False
        await query.edit_message_text(text="Cancelled.")
        return

    if "brightness" in query.message.text:

        BRIGHTNESS = int(query.data)

        await query.edit_message_text(text=f"Screen brightness set to: {query.data}")

        # -- resend to display
        user_folder = get_user_folder_from_query(query)
        original_file_path = os.path.join(user_folder, INPUT_FP)
        os.system(
            f"python scripts/measure/remote_display.py -cn {CONFIG_FN} fp={original_file_path} display.brightness={BRIGHTNESS} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
        )
        await query.edit_message_text(text=f"Image sent to display with brightness {BRIGHTNESS}.")
        # await update.message.reply_text("Image sent to display.", reply_to_message_id=update.message.message_id)

    elif "exposure" in query.message.text:

        EXPOSURE = float(query.data)
        await query.edit_message_text(text=f"Exposure set to {EXPOSURE} seconds.")

    # TODO not working with mask
    algo = DEFAULT_ALGO
    # send query instead of update as it has the message data
    await take_picture_and_reconstruct(update, context, algo, query=query)
    BUSY = False


async def brightness_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    Set brightness, re-capture, and reconstruct.
    """

    # check INPUT_FP exists
    user_folder = get_user_folder(update)
    original_file_path = os.path.join(user_folder, INPUT_FP)
    if not os.path.exists(original_file_path):
        await update.message.reply_text(
            "Please set an image first.", reply_to_message_id=update.message.message_id
        )
        return

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    vals = [20, 40, 60, 80, 100]
    vals.remove(BRIGHTNESS)
    keyboard = [
        [
            InlineKeyboardButton(f"{vals[0]}", callback_data=f"{vals[0]}"),
            InlineKeyboardButton(f"{vals[1]}", callback_data=f"{vals[1]}"),
        ],
        [
            InlineKeyboardButton(f"{vals[2]}", callback_data=f"{vals[2]}"),
            InlineKeyboardButton(f"{vals[3]}", callback_data=f"{vals[3]}"),
        ],
        [
            InlineKeyboardButton("Cancel", callback_data="Cancel"),
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"Please specify a value for the screen brightness. Current value is {BRIGHTNESS}",
        # reply_to_message_id=update.message.message_id,
        # reply_markup=ReplyKeyboardMarkup(
        #     reply_keyboard, resize_keyboard=True, one_time_keyboard=True, is_persistent=False, input_field_placeholder=f"Screen brightness value (current={BRIGHTNESS})."
        # ),
        reply_markup=reply_markup,
    )


async def exposure_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    Set exposure, re-capture, and reconstruct.
    """

    # # check INPUT_FP exists
    # user_folder = get_user_folder(update)
    # original_file_path = os.path.join(user_folder, INPUT_FP)
    # if not os.path.exists(original_file_path):
    #     await update.message.reply_text(
    #         "Please set an image first.", reply_to_message_id=update.message.message_id
    #     )
    #     return

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    # -- phase mask
    vals = {0.02: "very low", 0.04: "low", 0.06: "medium", 0.08: "high", 0.1: "very high"}
    # # -- tape based
    # vals = {0.02: "very low", 0.035: "low", 0.05: "medium", 0.065: "high", 0.08: "very high"}
    # # -- digicam
    # vals = {0.25: "very low", 0.5: "low", 0.75: "medium", 1: "high", 1.25: "very high"}

    if EXPOSURE in vals:
        del vals[EXPOSURE]
    keys = list(vals.keys())
    keyboard = [
        [
            InlineKeyboardButton(f"{vals[keys[0]]} ({keys[0]})", callback_data=f"{keys[0]}"),
            InlineKeyboardButton(f"{vals[keys[1]]} ({keys[1]})", callback_data=f"{keys[1]}"),
        ],
        [
            InlineKeyboardButton(f"{vals[keys[2]]} ({keys[2]})", callback_data=f"{keys[2]}"),
            InlineKeyboardButton(f"{vals[keys[3]]} ({keys[3]})", callback_data=f"{keys[3]}"),
        ],
        [
            InlineKeyboardButton("Cancel", callback_data="Cancel"),
        ],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"Please specify a value for the camera exposure. Current value is ({EXPOSURE} seconds).",
        reply_markup=reply_markup,
    )


async def emoji(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    global BUSY, EXPOSURE

    EXPOSURE = 0.7

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    # create image from emoji
    text = update.message.text
    size = 15
    with Image.new("RGB", (size, size), (0, 0, 0)) as image:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf", size, encoding="unic"
        )

        with Pilmoji(image) as pilmoji:
            pilmoji.text((0, 0), text.strip(), (0, 0, 0), font, align="center")

        # save image
        user_folder = get_user_folder(update)
        original_file_path = os.path.join(user_folder, INPUT_FP)
        image.save(original_file_path)

    # display
    vshift = -20
    brightness = 80
    os.system(
        f"python scripts/measure/remote_display.py -cn {CONFIG_FN} fp={original_file_path} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME} display.vshift={vshift} display.brightness={brightness}"
    )
    await update.message.reply_text(
        f"Image sent to display with brightness {brightness}.",
        reply_to_message_id=update.message.message_id,
    )

    await take_picture_and_reconstruct(update, context, DEFAULT_ALGO)
    BUSY = False


async def not_running_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "The bot is currently not running. If you want to try it out, please contact the admin.",
        reply_to_message_id=update.message.message_id,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="telegram_demo")
def main(config) -> None:
    """Start the bot."""

    global TOKEN, WHITELIST_USERS, RPI_USERNAME, RPI_HOSTNAME, RPI_LENSED_USERNAME, RPI_LENSED_HOSTNAME, CONFIG_FN, TIME_OFFSET
    global DEFAULT_ALGO, ALGO_TEXT, HELP_TEXT, supported_algos, supported_input
    global OVERLAY_ALPHA, OVERLAY_1, OVERLAY_2, OVERLAY_3, FILES_CAPTURE_CONFIG
    global PSF_FP, BACKGROUND_FP, MASK_PARAM, SETUP_FP
    global VSHIFT, IMAGE_RES

    TOKEN = config.token
    TIME_OFFSET = config.time_offset

    WHITELIST_USERS = config.whitelist
    if WHITELIST_USERS is None:
        WHITELIST_USERS = []

    if config.setup_fp is not None:
        SETUP_FP = config.setup_fp
        assert os.path.exists(SETUP_FP)

    RPI_USERNAME = config.rpi_username
    RPI_HOSTNAME = config.rpi_hostname
    RPI_LENSED_USERNAME = config.rpi_lensed_username
    RPI_LENSED_HOSTNAME = config.rpi_lensed_hostname
    CONFIG_FN = config.config_name
    DEFAULT_ALGO = config.default_algo

    OVERLAY_ALPHA = config.overlay.alpha
    OVERLAY_1 = config.overlay.img1
    OVERLAY_2 = config.overlay.img2
    OVERLAY_3 = config.overlay.img3

    if OVERLAY_1 is not None:
        assert os.path.exists(OVERLAY_1.fp)
    if OVERLAY_2 is not None:
        assert os.path.exists(OVERLAY_2.fp)
    if OVERLAY_3 is not None:
        assert os.path.exists(OVERLAY_3.fp)

    if config.supported_inputs is not None:
        supported_input = config.supported_inputs
    FILES_CAPTURE_CONFIG = config.files

    input_commands = ["/" + input for input in supported_input]
    HELP_TEXT = (
        "Through this bot, you can send a photo to the lensless camera setup in our lab at EPFL (shown above). "
        "The photo will be:\n\n1. Displayed on a screen.\n2. Our lensless camera will "
        "take a picture.\n3. A reconstruction will be sent back through the bot.\n4. "
        "The raw data will also be sent back."
        # "\n\nIf you do not feel comfortable sending one "
        # f"of your own pictures, you can use the {input_commands} commands to set "
        # "the image on the display with one of our inputs. Or even send an emoij 😎"
        f"\n\n⚠️ Try one of the {input_commands} commands to use images we've configured. "
        # "Or even send an emoji 😎 "
        "You can also send your own image (but brightness/exposure may need to be adjusted)."
        "\n\nAll previous data is overwritten "
        "when a new image is sent, and everything is deleted when the process running on the "
        "server is shut down."
    )

    if config.supported_algos is not None:
        supported_algos = config.supported_algos

    algo_commands = ["/" + algo for algo in supported_algos]
    ALGO_TEXT = (
        f"By default, the reconstruction is done with /{DEFAULT_ALGO}, but you "
        "can specify the algorithm (on the last measurement) with the corresponding "
        f"command: {algo_commands}."
        # "\n\nAll provided algorithms require an estimate of the point spread function (PSF). "
        # "Each user has their unique mask pattern according to the Telegram ID. "
        # "\n\n⚠️ After doing a measurement/reconstruction, you can try running /random_mask "
        # "to see what would be the reconstruction if you use a different (wrong) mask, "
        # "as if someone (like a hacker!) were trying to decode your data with a different mask."
        # "\n\nAll provided algorithms require an estimate of the point spread function (PSF). "
        # "You can measure a (proxy) PSF with /psf (a point source like "
        # "image will be displayed on the screen). "
        # "In practice, we measure the PSF with single white LED. The used PSF is sent also sent "
        # "back with the /psf command."
        "\n\nMore info: go.epfl.ch/lensless"
    )

    # make output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # load and downsample PSF beforehand
    if config.psf is not None:

        if "fp" in config.psf:
            from lensless.utils.io import save_image

            psf, bg = load_psf(
                config.psf.fp, downsample=config.psf.downsample, return_float=True, return_bg=True
            )
            # save to demo folder
            PSF_FP = os.path.join(OUTPUT_FOLDER, "psf.png")
            save_image(psf[0], PSF_FP)

            # save with gamma correction
            psf_gamma = psf[0] / np.max(psf[0])
            if config.gamma > 1:
                from lensless.utils.image import gamma_correction

                psf_gamma = gamma_correction(psf_gamma, gamma=config.gamma)
            save_image(psf_gamma, PSF_FP_GAMMA)

            # save background array
            BACKGROUND_FP = os.path.join(OUTPUT_FOLDER, "psf_bg.npy")
            np.save(BACKGROUND_FP, bg)
        elif "device" in config.psf:
            # programmable mask
            MASK_PARAM = config.psf

    # Create the Application and pass it your bot's token.
    assert TOKEN is not None
    application = Application.builder().token(TOKEN).build()

    if not config.idle:

        assert RPI_USERNAME is not None
        assert RPI_HOSTNAME is not None

        check_username_hostname(RPI_USERNAME, RPI_HOSTNAME)
        if RPI_LENSED_USERNAME is not None or RPI_LENSED_HOSTNAME is not None:
            check_username_hostname(RPI_LENSED_USERNAME, RPI_LENSED_HOSTNAME)

        # on different commands - answer in Telegram
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("notbusy", remove_busy_flag))

        for file_input in supported_input:
            assert file_input in FILES_CAPTURE_CONFIG.keys()
            application.add_handler(CommandHandler(file_input, file_input_command, block=False))

        # different algorithms
        application.add_handler(CommandHandler("fista", fista, block=False))
        application.add_handler(CommandHandler("admm", admm, block=False))
        application.add_handler(CommandHandler("unrolled", unrolled, block=False))
        application.add_handler(CommandHandler("unet", unet, block=False))

        # photo input
        application.add_handler(
            MessageHandler(filters.PHOTO & ~filters.COMMAND, photo, block=False)
        )

        # brightness input
        application.add_handler(CommandHandler("brightness", brightness_command, block=False))
        application.add_handler(CallbackQueryHandler(button))

        # exposure input
        application.add_handler(CommandHandler("exposure", exposure_command, block=False))
        application.add_handler(CallbackQueryHandler(button))

        # emoji input
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, emoji, block=False))

        if MASK_PARAM is not None:
            application.add_handler(CommandHandler("random_mask", random_mask, block=False))
        else:
            # to dim for measuring PSF of DigiCam?
            application.add_handler(CommandHandler("psf", psf_command, block=False))

        # Run the bot until the user presses Ctrl-C
        application.run_polling()

    else:

        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(None, not_running_command))

        # Run the bot until the user presses Ctrl-C
        application.run_polling()

    # delete non-empty folder
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
