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

# for displaying emojis
from emoji import EMOJI_DATA
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
from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup


TOKEN = None
RPI_USERNAME = None
RPI_HOSTNAME = None
RPI_LENSED_USERNAME = None
RPI_LENSED_HOSTNAME = None

OVERLAY_TOPLEFT = None  # e.g. logo of event/company
OVERLAY_TOPRIGHT = None
OVERLAY_BOTTOMLEFT = None
OVERLAY_BOTTOMRIGHT = None

PSF_FP = None
BACKGROUND_FP = None

INPUT_FP = "user_photo.jpg"
RAW_DATA_FP = "raw_data.png"
OUTPUT_FOLDER = "demo_lensless_recon"
BUSY = False
supported_algos = ["fista", "admm", "unrolled"]
supported_input = ["mnist", "thumb"]
DEFAULT_ALGO = "unrolled"
TIMEOUT = 1 * 60  # 10 minutes
ASYNC_PAUSE = 1
BRIGHTNESS = 100


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
    name = update.message.from_user.full_name.replace(" ", "-")
    user_subfolder = f"{update.message.from_user.id}_{name}"
    return os.path.join(OUTPUT_FOLDER, user_subfolder)


def get_user_folder_from_query(query):
    name = query.message.chat.full_name.replace(" ", "-")
    user_subfolder = f"{query.message.chat.id}_{name}"
    return os.path.join(OUTPUT_FOLDER, user_subfolder)


async def check_incoming_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global BUSY

    # print user
    print("User: ", update.message.from_user)

    # create folder for user
    user_folder = get_user_folder(update)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    if BUSY:
        return "System is busy. Please wait for the current job to finish and try again."

    # if message from a while ago, ignore
    utc = pytz.UTC
    now = utc.localize(datetime.now())
    message_time = update.message.date

    diff = (now - message_time).total_seconds()
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

    BUSY = True
    return


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! Through this bot, you can send a photo to our lensless camera setup at EPFL. The photo will be displayed on a screen, our lensless camera will take a picture of it, a reconstruction will be sent back through the bot, and the raw data and a (lensed) picture of the setup will also be sent back. If you do not feel comfortable sending one of your own pictures, you can use the /mnist, /thumb, /face commands to set the image on the display with one of our inputs. All previous data is overwritten when a new image is sent, and everything is deleted when the process running on the server is shut down. You can measure a (proxy) PSF with /psf (a point source like image will be displayed on the screen). The reconstruction is done with the Unrolled ADMM algorithm by default, but you can specify the algorithm (on the currently displayed image) with the corresponding command: /fista, /admm, /unrolled, /unet. More info: go.epfl.ch/lensless",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Through this bot, you can send a photo to our lensless camera setup at EPFL. The photo will be:\n1. Displayed on a screen.\n2. Our lensless camera will take a picture.\n3. A reconstruction will be sent back through the bot.\n4. The raw data and a (lensed) picture of the setup will also be sent back.\n\nIf you do not feel comfortable sending one of your own pictures, you can use the /mnist, /thumb, /face commands to set the image on the display with one of our inputs. All previous data is overwritten when a new image is sent, and everything is deleted when the process running on the server is shut down.\n\nYou can measure a (proxy) PSF with /psf (a point source like image will be displayed on the screen).\n\nThe reconstruction is done with the Unrolled ADMM algorithm by default, but you can specify the algorithm (on the currently displayed image) with the corresponding command: /fista, /admm, /unrolled, /unet.\n\nMore info: go.epfl.ch/lensless"
    )


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

    global BUSY

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
            f"python scripts/remote_display.py fp={original_file_path} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
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
            "Taking picture...", reply_to_message_id=query.message.message_id
        )
        os.system(
            f"python scripts/remote_capture.py plot=False rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME} output={user_subfolder}"
        )
    else:
        user_subfolder = get_user_folder(update)
        await update.message.reply_text(
            "Taking picture...", reply_to_message_id=update.message.message_id
        )
        os.system(
            f"python scripts/remote_capture.py plot=False rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME} output={user_subfolder}"
        )


async def reconstruct(update: Update, context: ContextTypes.DEFAULT_TYPE, algo, query=None) -> None:

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
            f"python scripts/recon/demo.py plot=False recon.algo={algo} output={user_subfolder} camera.psf={PSF_FP} recon.downsample=1 camera.background={BACKGROUND_FP}"
        )
    else:
        os.system(
            f"python scripts/recon/demo.py plot=False recon.algo={algo} output={user_subfolder}"
        )

    # -- send back, with watermark if provided
    if (
        OVERLAY_BOTTOMLEFT is not None
        or OVERLAY_TOPRIGHT is not None
        or OVERLAY_BOTTOMRIGHT is not None
        or OVERLAY_TOPLEFT is not None
    ):

        alpha = 60

        reconstructed_path = os.path.join(user_subfolder, "reconstructed.png")

        img1 = Image.open(reconstructed_path)
        img1 = img1.convert("RGBA")

        # first overlay
        if OVERLAY_TOPLEFT is not None:
            img2 = Image.open(OVERLAY_TOPLEFT)
            img2 = img2.convert("RGBA")
            img2.putalpha(alpha)
            new_width = int(img1.width * 0.7)
            img2 = img2.resize((new_width, int(new_width * img2.height / img2.width)))

        if OVERLAY_BOTTOMLEFT is not None:
            img3 = Image.open(OVERLAY_BOTTOMLEFT)
            img3 = img3.convert("RGBA")
            img3.putalpha(alpha)
            new_width = int(img1.width * 0.2)
            img3 = img3.resize((new_width, int(new_width * img3.height / img3.width)))

        if OVERLAY_BOTTOMRIGHT is not None:
            img4 = Image.open(OVERLAY_BOTTOMRIGHT)
            img4 = img4.convert("RGBA")
            img4.putalpha(alpha)
            new_width = int(img1.width * 0.23)
            img4 = img4.resize((new_width, int(new_width * img4.height / img4.width)))

        # overlay
        # img1.paste(img2, (20,0), mask = img2)
        # img1.paste(img3, (10, img1.height - img3.height - 10), mask = img3)
        # img1.paste(img4, (img1.width - img4.width, img1.height - img4.height - 10), mask = img4)
        img1.paste(img2, (20, 0), mask=img2)
        img1.paste(img3, (img2.width + 35, 25), mask=img3)
        img1.paste(img4, (img2.width + 27, 25 + img3.height), mask=img4)
        OUTPUT_FP = os.path.join(user_subfolder, "reconstructed_overlay.png")
        img1.convert("RGB").save(OUTPUT_FP)

        # return photo
        await update.message.reply_photo(
            OUTPUT_FP,
            caption=f"Reconstruction ({algo})",
            reply_to_message_id=update.message.message_id,
        )

    else:
        OUTPUT_FP = os.path.join(user_subfolder, "reconstructed.png")
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
    else:
        user_subfolder = get_user_folder(update)

    await take_picture(update, context, query=query)
    await reconstruct(update, context, algo, query=query)

    # # -- reconstruct
    # await update.message.reply_text(f"Reconstructing with {algo}...", reply_to_message_id=update.message.message_id)
    # os.system(f"python scripts/recon/demo.py plot=False recon.algo={algo}")
    # OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
    # # await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})", reply_to_message_id=update.message.message_id)
    # await update.message.reply_photo(OUTPUT_FP, caption=f"Reconstruction ({algo})", reply_to_message_id=update.message.message_id)
    # # img = np.array(Image.open(OUTPUT_FP))
    # # await update.message.reply_text("Output resolution: " + str(img.shape))

    # -- send picture of raw measurement
    OUTPUT_FP = os.path.join(user_subfolder, "raw_data_8bit.png")
    if query is not None:
        await query.message.reply_photo(
            OUTPUT_FP, caption="Raw measurement", reply_to_message_id=query.message.message_id
        )
    else:
        await update.message.reply_photo(
            OUTPUT_FP, caption="Raw measurement", reply_to_message_id=update.message.message_id
        )

    # -- send picture of setup (lensed)
    if RPI_LENSED_HOSTNAME is not None and RPI_LENSED_USERNAME is not None:
        os.system(
            f"python scripts/remote_capture.py rpi.username={RPI_LENSED_USERNAME} rpi.hostname={RPI_LENSED_HOSTNAME} plot=False capture.bayer=False capture.down=8 capture.raw_data_fn=lensed capture.awb_gains=null"
        )
        OUTPUT_FP = os.path.join(user_subfolder, "lensed.png")
        if query is not None:
            await query.message.reply_photo(
                OUTPUT_FP, caption="Picture of setup", reply_to_message_id=query.message.message_id
            )
        else:
            await update.message.reply_photo(
                OUTPUT_FP, caption="Picture of setup", reply_to_message_id=update.message.message_id
            )


async def mnist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    1. Use one of the input images
    2. Send to display
    3. Capture measurement
    4. Reconstruct
    """

    global BUSY

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    algo = DEFAULT_ALGO
    vshift = -10
    brightness = 100

    # copy image to INPUT_FP
    user_folder = get_user_folder(update)
    original_file_path = os.path.join(user_folder, INPUT_FP)
    os.system(f"cp data/original/mnist_3.png {original_file_path}")

    # -- send to display
    os.system(
        f"python scripts/remote_display.py fp={original_file_path} display.vshift={vshift} display.brightness={brightness} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text(
        f"Image sent to display with brightness {brightness}.",
        reply_to_message_id=update.message.message_id,
    )

    await take_picture_and_reconstruct(update, context, algo)
    BUSY = False


async def thumb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    1. Use one of the input images
    2. Send to display
    3. Capture measurement
    4. Reconstruct
    """

    global BUSY

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    algo = DEFAULT_ALGO
    vshift = -10
    brightness = 80

    # copy image to INPUT_FP
    user_folder = get_user_folder(update)
    original_file_path = os.path.join(user_folder, INPUT_FP)
    os.system(f"cp data/original/thumbs_up.png {original_file_path}")

    # -- send to display
    os.system(
        f"python scripts/remote_display.py fp={original_file_path} display.vshift={vshift} display.brightness={brightness} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text(
        f"Image sent to display with brightness {brightness}.",
        reply_to_message_id=update.message.message_id,
    )

    await take_picture_and_reconstruct(update, context, algo)
    BUSY = False


async def face_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    """
    1. Use one of the input images
    2. Send to display
    3. Capture measurement
    4. Reconstruct
    """

    global BUSY

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    algo = DEFAULT_ALGO
    vshift = 0
    brightness = 100

    # copy image to INPUT_FP
    user_folder = get_user_folder(update)
    original_file_path = os.path.join(user_folder, INPUT_FP)
    os.system(f"cp data/original/face.jpg {original_file_path}")

    # -- send to display
    os.system(
        f"python scripts/remote_display.py fp={original_file_path} display.vshift={vshift} display.brightness={brightness} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
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
        f"python scripts/remote_display.py display.psf={psf_size} display.vshift={vshift} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await update.message.reply_text(
        f"PSF of {psf_size}x{psf_size} pixels set on display.",
        reply_to_message_id=update.message.message_id,
    )

    # -- measurement
    os.system(
        f"python scripts/remote_capture.py -cn demo_measure_psf rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    OUTPUT_FP = os.path.join(OUTPUT_FOLDER, "raw_data.png")
    await update.message.reply_photo(
        OUTPUT_FP,
        caption="PSF (zoom in to see caustic pattern)",
        reply_to_message_id=update.message.message_id,
    )
    BUSY = False


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and updates the message text."""

    global BRIGHTNESS
    global BUSY
    BUSY = True

    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()

    if query.data == "Cancel":
        BUSY = False
        await query.edit_message_text(text="Cancelled.")
        return

    BRIGHTNESS = int(query.data)

    await query.edit_message_text(text=f"Screen brightness set to: {query.data}")

    # -- resend to display
    user_folder = get_user_folder_from_query(query)
    original_file_path = os.path.join(user_folder, INPUT_FP)
    os.system(
        f"python scripts/remote_display.py fp={original_file_path} display.brightness={BRIGHTNESS} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME}"
    )
    await query.edit_message_text(text=f"Image sent to display with brightness {BRIGHTNESS}.")
    # await update.message.reply_text("Image sent to display.", reply_to_message_id=update.message.message_id)

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


async def emoji(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    global BUSY

    res = await check_incoming_message(update, context)
    if res is not None:
        await update.message.reply_text(res, reply_to_message_id=update.message.message_id)
        return

    # create image from emoji
    text = update.message.text
    size = 30
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
    vshift = -10
    brightness = 100
    os.system(
        f"python scripts/remote_display.py fp={original_file_path} rpi.username={RPI_USERNAME} rpi.hostname={RPI_HOSTNAME} display.vshift={vshift} display.brightness={brightness}"
    )
    await update.message.reply_text(
        f"Image sent to display with brightness {brightness}.",
        reply_to_message_id=update.message.message_id,
    )

    await take_picture_and_reconstruct(update, context, DEFAULT_ALGO)
    BUSY = False


# async def overlay_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

#     reconstructed_path = os.path.join(OUTPUT_FOLDER, "reconstructed.png")
#     if not os.path.exists(reconstructed_path):
#         await update.message.reply_text(f"Please reconstruct an image first.", reply_to_message_id=update.message.message_id)
#         return

#     img1 = Image.open(reconstructed_path)
#     img1 = img1.convert("RGBA")

#     # secondary image
#     img2 = Image.open(OVERLAY_IMG)
#     img2 = img2.convert("RGBA")
#     img2.putalpha(75)

#     # resize img2 to width of img1, while maintaining aspect ratio
#     new_width = int(img1.width * 0.8)
#     img2 = img2.resize((new_width , int(new_width * img2.height / img2.width)))

#     # overlay
#     img1.paste(img2, (20,0), mask = img2)
#     output_fp = os.path.join(OUTPUT_FOLDER, "reconstructed_overlay.png")
#     img1.convert('RGB').save(output_fp)

#     # return photo
#     await update.message.reply_photo(output_fp, reply_to_message_id=update.message.message_id)


async def not_running_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "The bot is currently not running. Please contact the admin /help.",
        reply_to_message_id=update.message.message_id,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="telegram_demo")
def main(config) -> None:
    """Start the bot."""

    global TOKEN, RPI_USERNAME, RPI_HOSTNAME, RPI_LENSED_USERNAME, RPI_LENSED_HOSTNAME
    global OVERLAY_BOTTOMLEFT, OVERLAY_BOTTOMRIGHT, OVERLAY_TOPLEFT, OVERLAY_TOPRIGHT
    global PSF_FP, BACKGROUND_FP

    TOKEN = config.token
    RPI_USERNAME = config.rpi_username
    RPI_HOSTNAME = config.rpi_hostname
    RPI_LENSED_USERNAME = config.rpi_lensed_username
    RPI_LENSED_HOSTNAME = config.rpi_lensed_hostname
    OVERLAY_BOTTOMRIGHT = config.overlay_bottom_right
    OVERLAY_BOTTOMLEFT = config.overlay_bottom_left
    OVERLAY_TOPRIGHT = config.overlay_top_right
    OVERLAY_TOPLEFT = config.overlay_top_left

    # make output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # load and downsample PSF beforehand
    if config.psf is not None:
        from lensless.io import load_psf, save_image

        psf, bg = load_psf(
            config.psf, downsample=config.downsample, return_float=True, return_bg=True
        )

        # save to demo folder
        PSF_FP = os.path.join(OUTPUT_FOLDER, "psf.png")
        save_image(psf[0], PSF_FP)

        # save background array
        BACKGROUND_FP = os.path.join(OUTPUT_FOLDER, "psf_bg.npy")
        np.save(BACKGROUND_FP, bg)

    # Create the Application and pass it your bot's token.
    assert TOKEN is not None
    application = Application.builder().token(TOKEN).build()

    if not config.idle:

        assert RPI_USERNAME is not None
        assert RPI_HOSTNAME is not None

        # on different commands - answer in Telegram
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("mnist", mnist_command, block=False))
        application.add_handler(CommandHandler("thumb", thumb_command, block=False))
        application.add_handler(CommandHandler("face", face_command, block=False))
        application.add_handler(CommandHandler("psf", psf_command, block=False))
        # application.add_handler(CommandHandler("brightness", brightness_command, block=False))

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

        # emoji input
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, emoji, block=False))

        # overlay images
        if OVERLAY_BOTTOMLEFT is not None:
            assert os.path.exists(OVERLAY_BOTTOMLEFT)
        if OVERLAY_BOTTOMRIGHT is not None:
            assert os.path.exists(OVERLAY_BOTTOMRIGHT)
        if OVERLAY_TOPLEFT is not None:
            assert os.path.exists(OVERLAY_TOPLEFT)
        if OVERLAY_TOPRIGHT is not None:
            assert os.path.exists(OVERLAY_TOPRIGHT)

        # # overlay input
        # if OVERLAY_IMG is not None:
        #     assert os.path.exists(OVERLAY_IMG)
        # application.add_handler(CommandHandler("overlay", overlay_command, block=False))

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
