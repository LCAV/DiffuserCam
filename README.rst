=============
LenslessPiCam
=============

.. image:: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
    :target: https://github.com/LCAV/LenslessPiCam
    :alt: GitHub page

.. image:: https://readthedocs.org/projects/lensless/badge/?version=latest
    :target: http://lensless.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://joss.theoj.org/papers/10.21105/joss.04747/status.svg
      :target: https://doi.org/10.21105/joss.04747
      :alt: DOI

.. image:: https://static.pepy.tech/badge/lensless
      :target: https://www.pepy.tech/projects/lensless
      :alt: Downloads

.. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://lensless.readthedocs.io/en/latest/examples.html
      :alt: notebooks

.. image:: https://img.shields.io/badge/Google_Slides-yellow
      :target: https://docs.google.com/presentation/d/1PcNhMfjATSwcpbHUMrmc88ciQmheBJ7alz8hel8xnGU/edit?usp=sharing
      :alt: slides

.. image:: https://huggingface.co/datasets/huggingface/badges/resolve/main/powered-by-huggingface-dark.svg
      :target: https://huggingface.co/bezzam
      :alt: huggingface


*A Hardware and Software Toolkit for Lensless Computational Imaging*
--------------------------------------------------------------------

.. image:: https://github.com/LCAV/LenslessPiCam/raw/main/scripts/recon/example.png
    :alt: Lensless imaging example
    :align: center


This toolkit has everything you need to perform imaging with a lensless camera.
The sensor in most examples is the `Raspberry Pi HQ <https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera>`__,
camera sensor as it is low cost (around 50 USD) and has a high resolution (12 MP).
The lensless encoder/mask used in most examples is either a piece of tape or a `low-cost LCD <https://www.adafruit.com/product/358>`__.
As **modularity** is a key feature of this toolkit, we try to support different sensors and/or lensless encoders.

The toolkit includes:

* Training scripts/configuration for various learnable, physics-informed reconstruction approaches, as shown `here <https://github.com/LCAV/LenslessPiCam/blob/main/configs/train#training-physics-informed-reconstruction-models>`__.
* Camera assembly tutorials (`link <https://lensless.readthedocs.io/en/latest/building.html>`__).
* Measurement scripts (`link <https://lensless.readthedocs.io/en/latest/measurement.html>`__).
* Dataset preparation and loading tools, with `Hugging Face <https://huggingface.co/bezzam>`__ integration (`slides <https://docs.google.com/presentation/d/18h7jTcp20jeoiF8dJIEcc7wHgjpgFgVxZ_bJ04W55lg/edit?usp=sharing>`__ on uploading a dataset to Hugging Face with `this script <https://github.com/LCAV/LenslessPiCam/blob/main/scripts/data/upload_dataset_huggingface.py>`__).
* `Reconstruction algorithms <https://lensless.readthedocs.io/en/latest/reconstruction.html>`__ (e.g. FISTA, ADMM, unrolled algorithms, trainable inversion, , multi-Wiener deconvolution network, pre- and post-processors).
* `Pre-trained models <https://github.com/LCAV/LenslessPiCam/blob/main/lensless/recon/model_dict.py>`__ that can be loaded from `Hugging Face <https://huggingface.co/bezzam>`__, for example in `this script <https://github.com/LCAV/LenslessPiCam/blob/main/scripts/recon/diffusercam_mirflickr.py>`__.
* Mask `design <https://lensless.readthedocs.io/en/latest/mask.html>`__ and `fabrication <https://lensless.readthedocs.io/en/latest/fabrication.html>`__ tools.
* `Simulation tools <https://lensless.readthedocs.io/en/latest/simulation.html>`__.
* `Evalutions tools <https://lensless.readthedocs.io/en/latest/evaluation.html>`__ (e.g. PSNR, LPIPS, SSIM, visualizations).
* `Demo <https://lensless.readthedocs.io/en/latest/demo.html#telegram-demo>`__ that can be run on Telegram!

Please refer to the `documentation <http://lensless.readthedocs.io>`__ for more details,
while an overview of example notebooks can be found `here <https://lensless.readthedocs.io/en/latest/examples.html>`__.

We've also written a few Medium articles to guide users through the process
of building the camera, measuring data with it, and reconstruction.
They are all laid out in `this post <https://medium.com/@bezzam/a-complete-lensless-imaging-tutorial-hardware-software-and-algorithms-8873fa81a660>`__.

Collection of lensless imaging research
---------------------------------------

The following works have been implemented in the toolkit:

Reconstruction algorithms:

* ADMM with total variation regularization and 3D support (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/recon/admm.py#L24>`__, `usage <https://github.com/LCAV/LenslessPiCam/blob/main/scripts/recon/admm.py>`__). [1]_
* Unrolled ADMM (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/recon/unrolled_admm.py#L20>`__, `usage <https://github.com/LCAV/LenslessPiCam/tree/main/configs/train#unrolled-admm>`__). [2]_
* Unrolled ADMM with compensation branch (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/recon/utils.py#L84>`__, `usage <https://github.com/LCAV/LenslessPiCam/tree/main/configs/train#compensation-branch>`__). [3]_
* Trainable inversion from Flatnet (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/recon/trainable_inversion.py#L11>`__, `usage <https://github.com/LCAV/LenslessPiCam/tree/main/configs/train#trainable-inversion>`__). [4]_
* Multi-Wiener deconvolution network (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/recon/multi_wiener.py#L87>`__, `usage <https://github.com/LCAV/LenslessPiCam/tree/main/configs/train#multi-wiener-deconvolution-network>`__). [5]_
* SVDeconvNet (for learning multi-PSF deconvolution) from PhoCoLens (`source code <https://github.com/LCAV/LenslessPiCam/blob/main/lensless/recon/sv_deconvnet.py#L42>`__, `usage <https://github.com/LCAV/LenslessPiCam/tree/main/configs/train#multi-psf-camera-inversion>`__). [6]_
* Incorporating pre-processor (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/recon/trainable_recon.py#L52>`__). [7]_
* Accounting for external illumination(`source code 1 <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/recon/trainable_recon.py#L64>`__, `source code 2 <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/scripts/recon/train_learning_based.py#L458>`__, `usage <https://github.com/LCAV/LenslessPiCam/tree/main/configs/train#multilens-under-external-illumination>`__). [8]_ 

Camera / mask design:

* Fresnel zone aperture mask pattern (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/hardware/mask.py#L823>`__). [9]_ 
* Coded aperture mask pattern (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/hardware/mask.py#L288>`__). [10]_
* Near-field Phase Retrieval for designing a high-contrast phase mask (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/hardware/mask.py#L706>`__). [11]_
* LCD-based camera, i.e. DigiCam (`source code <https://github.com/LCAV/LenslessPiCam/blob/d0261b4bc79ef05228b135e6898deb4f7793d1aa/lensless/hardware/trainable_mask.py#L117>`__). [7]_ 

Datasets (hosted on Hugging Face and downloaded via their API):

* DiffuserCam Lensless MIR Flickr dataset (copy on `Hugging Face <https://huggingface.co/datasets/bezzam/DiffuserCam-Lensless-Mirflickr-Dataset-NORM>`__). [2]_
* TapeCam MIR Flickr (`Hugging Face <https://huggingface.co/datasets/bezzam/TapeCam-Mirflickr-25K>`__). [7]_ 
* DigiCam MIR Flickr (`Hugging Face <https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-SingleMask-25K>`__). [7]_
* DigiCam MIR Flickr with multiple mask patterns (`Hugging Face <https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-25K>`__). [7]_ 
* DigiCam CelebA (`Hugging Face <https://huggingface.co/datasets/bezzam/DigiCam-CelebA-26K>`__). [7]_
* MultiFocal mask MIR Flickr under external illumination (`Hugging Face <https://huggingface.co/datasets/Lensless/MultiLens-Mirflickr-Ambient>`__). [8]_ Mask fabricated by [12]_


Setup 
-----

If you are just interested in using the reconstruction algorithms and 
plotting / evaluation tools you can install the package via ``pip``:

.. code:: bash

   pip install lensless


For plotting, you may also need to install
`Tk <https://stackoverflow.com/questions/5459444/tkinter-python-may-not-be-configured-for-tk>`__.


For performing measurements, the expected workflow is to have a local 
computer which interfaces remotely with a Raspberry Pi equipped with 
the HQ camera sensor (or V2 sensor). Instructions on building the camera
can be found `here <https://lensless.readthedocs.io/en/latest/building.html>`__.

The software from this repository has to be installed on **both** your
local machine and the Raspberry Pi. Note that we recommend using
Python 3.11, as some Python library versions may not be available with 
earlier versions of Python. Moreover, its `end-of-life <https://endoflife.date/python>`__ 
is Oct 2027.

*Local machine setup*
=====================

Below are commands that worked for our configuration (Ubuntu 22.04.5 LTS), 
but there are certainly other ways to download a repository and
install the library locally.

Note that ``(lensless)`` is a convention to indicate that the virtual
environment is activated. After activating your virtual environment, you only
have to copy the command after ``(lensless)``.

.. code:: bash

   # download from GitHub
   git clone git@github.com:LCAV/LenslessPiCam.git
   cd LenslessPiCam

   # create virtual environment (as of Oct 4 2023, rawpy is not compatible with Python 3.12)
   # -- using conda
   conda create -n lensless python=3.11
   conda activate lensless

   # -- OR venv
   python3.11 -m venv lensless_env
   source lensless_env/bin/activate

   # install package
   (lensless) pip install -e .

   # extra dependencies for local machine for plotting/reconstruction
   (lensless) pip install -r recon_requirements.txt

   # pre-commit hooks for code formatting
   (lensless) pip install pre-commit black
   (lensless) pre-commit install

   # (optional) try reconstruction on local machine
   (lensless) python scripts/recon/admm.py

   # (optional) try reconstruction on local machine with GPU
   (lensless) python scripts/recon/admm.py -cn pytorch


Note (25-04-2023): for using the :py:class:`~lensless.recon.apgd.APGD` reconstruction method based on Pycsou
(now `Pyxu <https://github.com/matthieumeo/pyxu>`__), a specific commit has 
to be installed (as there was no release at the time of implementation):

.. code:: bash

   pip install git+https://github.com/matthieumeo/pycsou.git@38e9929c29509d350a7ff12c514e2880fdc99d6e

If PyTorch is installed, you will need to be sure to have PyTorch 2.0 or higher, 
as Pycsou is not compatible with earlier versions of PyTorch. Moreover, 
Pycsou requires Python within 
`[3.9, 3.11) <https://github.com/matthieumeo/pycsou/blob/v2-dev/setup.cfg#L28>`__.

Moreover, ``numba`` (requirement for Pycsou V2) may require an older version of NumPy:

.. code:: bash

   pip install numpy==1.23.5

*Raspberry Pi setup*
====================

After `flashing your Raspberry Pi with SSH enabled <https://medium.com/@bezzam/setting-up-a-raspberry-pi-without-a-monitor-headless-9a3c2337f329>`__, 
you need to set it up for `passwordless access <https://medium.com/@bezzam/headless-and-passwordless-interfacing-with-a-raspberry-pi-ssh-453dd75154c3>`__. 
Do not set a password for your SSH key pair, as this will not work with the
provided scripts.

On the Raspberry Pi, you can then run the following commands (from the ``home`` 
directory):

.. code:: bash

   # dependencies
   sudo apt-get install -y libimage-exiftool-perl libatlas-base-dev \
   python3-numpy python3-scipy python3-opencv
   sudo pip3 install -U virtualenv

   # download from GitHub
   git clone git@github.com:LCAV/LenslessPiCam.git

   # install in virtual environment
   cd LenslessPiCam
   virtualenv --system-site-packages -p python3 lensless_env
   source lensless_env/bin/activate
   pip install --no-deps -e .
   pip install -r rpi_requirements.txt

   # test on-device camera capture (after setting up the camera)
   (lensless_env) python scripts/measure/on_device_capture.py

You may still need to manually install ``numpy`` and/or ``scipy`` with ``pip`` in case libraries (e.g. ``libopenblas.so.0``) cannot be detected.
   

Acknowledgements
----------------

The idea of building a lensless camera from a Raspberry Pi and a piece of 
tape comes from Prof. Laura Waller's group at UC Berkeley. So a huge kudos 
to them for the idea and making tools/code/data available! Below is some of 
the work that has inspired this toolkit:

* `Build your own DiffuserCam tutorial <https://waller-lab.github.io/DiffuserCam/tutorial>`__.
* `DiffuserCam Lensless MIR Flickr dataset <https://waller-lab.github.io/LenslessLearning/dataset.html>`__ [2]_. 

A few students at EPFL have also contributed to this project:

* Julien Sahli: support and extension of algorithms for 3D.
* Yohann Perron: unrolled algorithms for reconstruction.
* Aaron Fargeon: mask designs.
* Rein Bentdal and David Karoubi: mask fabrication with 3D printing.
* Stefan Peters: imaging under external illumination.

We also thank the Swiss National Science Foundation for funding this project through the `Open Research Data (ORD) program <https://ethrat.ch/en/eth-domain/open-research-data/>`__.

Citing this work
----------------

If you use this toolkit in your own research, please cite the following:

::

   @article{Bezzam2023,
      doi = {10.21105/joss.04747},
      url = {https://doi.org/10.21105/joss.04747},
      year = {2023},
      publisher = {The Open Journal},
      volume = {8},
      number = {86},
      pages = {4747},
      author = {Eric Bezzam and Sepand Kashani and Martin Vetterli and Matthieu Simeoni},
      title = {LenslessPiCam: A Hardware and Software Platform for Lensless Computational Imaging with a Raspberry Pi},
      journal = {Journal of Open Source Software}
   }


The following papers have contributed new approaches to the field of lensless imaging:

* Introducing pre-processor component as part of modular reconstruction (`IEEE Transactions on Computational Imaging <https://arxiv.org/abs/2502.01102>`__ and `IEEE International Conference on Image Processing (ICIP) 2024 <https://arxiv.org/abs/2403.00537>`__):

::

   @ARTICLE{10908470,
      author={Bezzam, Eric and Perron, Yohann and Vetterli, Martin},
      journal={IEEE Transactions on Computational Imaging}, 
      title={Towards Robust and Generalizable Lensless Imaging With Modular Learned Reconstruction}, 
      year={2025},
      volume={11},
      number={},
      pages={213-227},
      keywords={Training;Wiener filters;Computational modeling;Transfer learning;Computer architecture;Cameras;Transformers;Software;Software measurement;Image reconstruction;Lensless imaging;modularity;robustness;generalizability;programmable mask;transfer learning},
      doi={10.1109/TCI.2025.3539448}
   }
   
   @INPROCEEDINGS{10647433,
      author={Perron, Yohann and Bezzam, Eric and Vetterli, Martin},
      booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
      title={A Modular and Robust Physics-Based Approach for Lensless Image Reconstruction}, 
      year={2024},
      volume={},
      number={},
      pages={3979-3985},
      keywords={Training;Multiplexing;Pipelines;Noise;Cameras;Robustness;Reproducibility of results;Lensless imaging;modular reconstruction;end-to-end optimization},
      doi={10.1109/ICIP51287.2024.10647433}
   }


* Lensless imaging under external illumination (`IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2025 <https://arxiv.org/abs/2502.01102>`__):

::

   @INPROCEEDINGS{10888030,
      author={Bezzam, Eric and Peters, Stefan and Vetterli, Martin},
      booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
      title={Let There Be Light: Robust Lensless Imaging Under External Illumination With Deep Learning}, 
      year={2025},
      volume={},
      number={},
      pages={1-5},
      keywords={Source separation;Noise;Lighting;Interference;Reconstruction algorithms;Cameras;Optics;Speech processing;Image reconstruction;Standards;lensless imaging;ambient lighting;external illumination;background subtraction;learned reconstruction},
      doi={10.1109/ICASSP49660.2025.10888030}
   }

References
----------

.. [1] Antipa, N., Kuo, G., Heckel, R., Mildenhall, B., Bostan, E., Ng, R., & Waller, L. (2017). DiffuserCam: lensless single-exposure 3D imaging. Optica, 5(1), 1-9.
.. [2] Monakhova, K., Yurtsever, J., Kuo, G., Antipa, N., Yanny, K., & Waller, L. (2019). Learned reconstructions for practical mask-based lensless imaging. Optics express, 27(20), 28075-28090.
.. [3] Zeng, T., & Lam, E. Y. (2021). Robust reconstruction with deep learning to handle model mismatch in lensless imaging. IEEE Transactions on Computational Imaging, 7, 1080-1092.
.. [4] Khan, S. S., Sundar, V., Boominathan, V., Veeraraghavan, A., & Mitra, K. (2020). Flatnet: Towards photorealistic scene reconstruction from lensless measurements. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(4), 1934-1948.
.. [5] Li, Y., Li, Z., Chen, K., Guo, Y., & Rao, C. (2023). MWDNs: reconstruction in multi-scale feature spaces for lensless imaging. Optics Express, 31(23), 39088-39101.
.. [6] Cai, X., You, Z., Zhang, H., Gu, J., Liu, W., & Xue, T. (2024). Phocolens: Photorealistic and consistent reconstruction in lensless imaging. Advances in Neural Information Processing Systems, 37, 12219-12242.
.. [7] Bezzam, E., Perron, Y., & Vetterli, M. (2025). Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction. IEEE Transactions on Computational Imaging.
.. [8] Bezzam, E., Peters, S., & Vetterli, M. (2024). Let there be light: Robust lensless imaging under external illumination with deep learning. IEEE International Conference on Acoustics, Speech and Signal Processing.
.. [9] Wu, J., Zhang, H., Zhang, W., Jin, G., Cao, L., & Barbastathis, G. (2020). Single-shot lensless imaging with fresnel zone aperture and incoherent illumination. Light: Science & Applications, 9(1), 53.
.. [10] Asif, M. S., Ayremlou, A., Sankaranarayanan, A., Veeraraghavan, A., & Baraniuk, R. G. (2016). Flatcam: Thin, lensless cameras using coded aperture and computation. IEEE Transactions on Computational Imaging, 3(3), 384-397.
.. [11] Boominathan, V., Adams, J. K., Robinson, J. T., & Veeraraghavan, A. (2020). Phlatcam: Designed phase-mask based thin lensless camera. IEEE transactions on pattern analysis and machine intelligence, 42(7), 1618-1629.
.. [12] Lee, K. C., Bae, J., Baek, N., Jung, J., Park, W., & Lee, S. A. (2023). Design and single-shot fabrication of lensless cameras with arbitrary point spread functions. Optica, 10(1), 72-80.
