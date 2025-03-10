# Configuration file for the Sphinx documentation builder.
import datetime
import os
import sys
from unittest import mock


MOCK_MODULES = [
    "scipy",
    "scipy.signal",
    "scipy.linalg",
    "pycsou",
    "matplotlib",
    "matplotlib.pyplot",
    "scipy.fftpack",
    "cv2",
    "rawpy",
    "skimage.metrics",
    "lpips",
    "torchmetrics",
    "torchmetrics.image",
    "scipy.ndimage",
    "pycsou.abc",
    "pycsou.operator",
    "pycsou.operator.func",
    "pycsou.operator.linop",
    "pycsou.opt.solver",
    "pycsou.opt.stop",
    "pycsou.runtime",
    "pycsou.util",
    "pycsou.util.ptype",
    "PIL",
    "PIL.Image",
    "slm_controller",
    "slm_controller.hardware",
    "paramiko",
    "paramiko.ssh_exception",
    "perlin_numpy",
    "hydra",
    "hydra.utils",
    "scipy.special",
    "matplotlib.cm",
    "pyffs",
    "datasets",
    "huggingface_hub",
    "cadquery",
    "wandb",
    "einops",
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()
# -- Project information

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
from lensless import __version__

project = "LenslessPiCam"
copyright = f"{datetime.date.today().year}, Eric Bezzam"
author = "Eric Bezzam"
version = __version__
release = version

# -- General configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "NumPy [latest]": ("https://docs.scipy.org/doc/numpy/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("http://matplotlib.sourceforge.net/", None),
    "pycsou": ("https://matthieumeo.github.io/pycsou/html/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
# html_theme_options = {"navigation_depth": -1, "titles_only": True}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
