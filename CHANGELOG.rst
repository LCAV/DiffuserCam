Changelog
=========

All notable changes to `LenslessPiCam
<https://github.com/LCAV/LenslessPiCam>`_ will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`__.


Unreleased
----------

Added
~~~~~

- Option to set image resolution with ``hardware.utils.display`` function.
- Add utility for mask adapter generation in ``lenseless.hardware.fabrication``
- Option to add simulated background in ``util.dataset``
- Auxiliary of reconstructing output from pre-processor (not working).
- Option to set focal range for MultiLensArray.
- Optional to remove deadspace modelling for programmable mask.
- Compensation branch for unrolled ADMM: https://ieeexplore.ieee.org/abstract/document/9546648
- Multi-Wiener deconvolution network: https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-23-39088&id=541387
- Option to skip pre-processor and post-processor at inference time.
- Option to set difference learning rate schedules, e.g. ADAMW, exponential decay, Cosine decay with warmup.
- Various augmentations for training: random flipping, random rotate, and random shifts. Latter two don't work well since new regions appear that throw off PSF/LSI modeling.
- HFSimulated object for simulating lensless data from ground-truth and PSF.
- Option to set cache directory for Hugging Face datasets.
- Option to initialize training with another model.
- Option to pass background image to ``utils.io.load_data``.
- Option to use background in ``lensless.eval.benchmark``. 
- Different techniques to use measured background: direct subtraction, learned subtraction, integrated subtraction, concatenated to input.
- Learnable background subtraction for classes that derive from ``lensless.recon.trainable_recon.TrainableReconstructionAlgorithm``.
- Integrated background subtraction object ``lensless.recon.integrated_background.IntegratedBackgroundSub``.
- Option to concatenate background to input to pre-processor.
- Add support for datasets with measured background to ``lensless.utils.dataset.HFDataset``.
- Add parameterize and peturb to evaluate model adaptation.
- PSF correction network.
- Option to add noise to input image or PSF (for robustness experiments).
- Learnable shift-variant forward model similar to PhoCoLens: https://phocolens.github.io/


Changed
~~~~~~~

- ``lensless.utils.dataset.HFDataset`` no longer inherits from ``lensless.utils.dataset.DualDataset``.

Bugfix
~~~~~~

- Wavelength input to ``compute_psf`` for Mask class.
- Computation of average metric in batches.
- Support for grayscale PSF for RealFFTConvolve2D.
- Calling model.eval() before inference, and model.train() before training.
- Normalization when computing reconstruction error.


1.0.7 - (2024-05-14)
--------------------

Added
~~~~~

- Script to upload measured datasets to Hugging Face: ``scripts/data/upload_dataset_huggingface.py``
- Pytorch support for simulating PSFs of masks.
- ``lensless.hardware.mask.MultiLensArray`` class for simulating multi-lens arrays.
- ``lensless.hardware.trainable_mask.TrainableCodedAperture`` class for training a coded aperture mask pattern.
- Support for other optimizers in ``lensless.utils.Trainer.set_optimizer``.
- ``lensless.utils.dataset.simulate_dataset`` for simulating a dataset given a mask/PSF.
- Support for training/testing with multiple mask patterns in the dataset.
- Multi-GPU support for training.
- Dataset which interfaces with Hugging Face (``lensless.utils.dataset.HFDataset``).
- DigiCam support for Telegram demo.
- DiffuserCamMirflickr Hugging Face API.
- Fallback for normalization if data not in 8bit range (``lensless.utils.io.save_image``).
- Add utilities for fabricating masks with 3D printing (``lensless.hardware.fabrication``).
- WandB support.
- Script for Mask adapter generation and update new mount in doc

Changed
~~~~~~~

- Dataset reconstruction script uses datasets from Hugging Face: ``scripts/recon/dataset.py``
- For trainable masks, set trainable parameters inside the child class.
- ``distance_sensor`` optional for ``lensless.hardware.mask.Mask``, e.g. don't need for fabrication.
- More intuitive interface for MURA for coded aperture (``lensless.hardware.mask.CodedAperture``), i.e. directly pass prime number.
- ``is_torch`` to ``use_torch`` in ``lensless.hardware.mask.Mask``
- ``self.height_map`` as characterization of phase masks (instead of phase pattern which can change for each wavelength)


Bugfix
~~~~~~

- ``lensless.hardware.trainable_mask.AdafruitLCD`` input handling.
- Local path for DRUNet download.
- APGD input handling (float32).
- Multimask handling.
- Passing shape to IRFFT so that it matches shape of input to RFFT.
- MLS mask creation (needed to rescale digits).

1.0.6 - (2024-02-21)
--------------------

Added
~~~~~

- Trainable reconstruction can return intermediate outputs (between pre- and post-processing).
- Auto-download for DRUNet model.
- ``utils.dataset.DiffuserCamMirflickr`` helper class for Mirflickr dataset.
- Option to crop section of image for computing loss when training unrolled.
- Option to learn color filter of RGB mask.
- Trainable mask for Adafruit LCD.
- Utility for capture image.
- Option to freeze/unfreeze/add pre- and post-processor components during training.
- Option to skip unrolled training and just use U-Net.
- Dataset objects for Adafruit LCD: measured CelebA and hardware-in-the-loop.
- Option to add auxiliary loss from output of camera inversion.
- Option to specify denoiser to iterative methods for plug-and-play.
- Model repository of trained models in ``lensless.recon.model_dict``.
- TrainableInversion component as in FlatNet.
- ``lensless.recon.utils.get_drunet_function_v2`` which doesn't normalize each color channel.
- Option to add noise to DiffuserCamMirflickr dataset.
- Option to initialize pre- and post-processor with components from another model.

Changed
~~~~~~~

- Better logic for saving best model. Based on desired metric rather than last epoch, and intermediate models can be saved.
- Optional normalization in ``utils.io.load_image``.

Bugfix
~~~~~~

- Support for unrolled reconstruction with grayscale, needed to copy to three channels for LPIPS.
- Fix bad train/test split for DiffuserCamMirflickr in unrolled training.
- Resize utility.
- Aperture, index to dimension conversion.
- Submodule imports.


1.0.5 - (2023-09-05)
--------------------

Added
~~~~~

- Sensor module.
- Single-script and Telegram demo.
- Link and citation for JOSS.
- Authors at top of source code files.
- Add paramiko as dependency for remote capture and display.
- Mask module, for CodedAperture (FlatCam), PhaseContour (PhlatCam), and FresnelZoneAperture.
- Script for measuring arbitrary dataset (from Raspberry Pi).
- Support for preprocessing and postprocessing, such as denoising, in ``TrainableReconstructionAlgorithm``. Both trainable and fix postprocessing can be used.
- Utilities to load a trained DruNet model for use as postprocessing in ``TrainableReconstructionAlgorithm``.
- Unified interface for dataset. See ``utils.dataset.DualDataset``.
- New simulated dataset compatible with new data format ([(batch_size), depth, width, height, color]). See ``utils.dataset.SimulatedFarFieldDataset``.
- New dataset for pair of original image and their measurement from a screen. See ``utils.dataset.MeasuredDataset`` and ``utils.dataset.MeasuredDatasetSimulatedOriginal``.
- Support for unrolled loading and inference in the script ``admm.py``.
- Tikhonov reconstruction for coded aperture measurements (MLS / MURA): numpy and Pytorch support.
- New ``Trainer`` class to train ``TrainableReconstructionAlgorithm`` with PyTorch.
- New ``TrainableMask`` and ``TrainablePSF`` class to train/fine-tune a mask from a dataset.
- New ``SimulatedDatasetTrainableMask`` class to train/fine-tune a mask for measurement.
- PyTorch support for ``lensless.utils.io.rgb2gray``.


Changed
~~~~~~~

- Simpler remote capture and display scripts with Hydra.
- Group source code into four modules: ``hardware``, ``recon``, ``utils``, ``eval``.
- Split scripts into subfolders.
- Displaying 3D reconstructions now shows projections on all three axis.


Bugfix
~~~~~~

- Fix overwriting of sensor parameters when downsampling.
- Displaying 3D reconstructions by summing values along axis would produce un-normalized values.

1.0.4 - (2023-06-14)
--------------------

Bugfix
~~~~~~

- Fix rendering of README on PyPI.


1.0.3 - (2023-06-14)
--------------------

Added
~~~~~

-  Documentation files and configuration, using Sphinx.
-  Implementations for ``autocorr2d`` and ``RealFFTConvolve2D``.
-  Benchmarking tool for ReconstructionAlgorithm
-  ``n_iter`` parameter for ReconstructionAlgorithm constructor, so don't need to pass to ``apply``.
-  Support of 3D reconstruction for Gradient Descent and APGD, with and without Pytorch.
-  Option to warm-start reconstruction algorithm with ``initial_est``.
-  TrainableReconstructionAlgorithm class inherited from ReconstructionAlgorithm and torch.module for use with pytorch autograd and optimizers.
-  Unrolled version of FISTA and ADMM as TrainableReconstructionAlgorithm with learnable parameters.
- ``train_learning_based.py`` script for training unrolled algorithms.
- ``benchmark_recon.py`` script for benchmarking and comparing reconstruction algorithms.
- Added ``reconstruction_error`` to ``ReconstructionAlgorithm`` .
- Added support for npy/npz image in load_image.

Changed
~~~~~~~

-  README.md to READ.rst for documentation.
-  CONTRIBUTING and CHANGELOG, to .rst for documentation.
-  Shorten README to separate contents in different pages of docs.
-  Fix typo in GradientDescent class name.
-  Updated to Pycsou V2, as ``pip install pycsou`` (Pycsou V1) may not work on some machines.
-  PSF are now always stored as 4D Data [depth, width, height, color], Data are stored as [(batch_size), depth, width, height, color] batch_size being optional. Grayscale data has a color axis of length 1 and 2D data has a depth axis of length 1.
-  Added batch support to RealFFTConvolve2D.
-  ``ReconstructionAlgorithm.update`` now take the number of the current iteration to allow for unrolled algorithms.
-  ``ReconstructionAlgorithm.apply`` now takes a reset parameter (default true) to automaticaly call reset.
-  Call to reset in ``ReconstructionAlgorithm.__init__`` is now optional (see reset parameter).
-  Make sure image estimate is reset when reset() is called, either to zeros/mean data or to self._initial_est if set.

Bugfix
~~~~~~

-  Loading grayscale PSFs would cause an dimension error when removing the background pixels.


1.0.2 - (2022-05-31)
--------------------

Added
~~~~~

-  Example of RGB reconstruction with complex-valued FFT: ``scripts/recon/apgd_pycsou.py``


Bugfix
~~~~~~

-  Possible shape mismatch when using the real-valued FFT: forward and
   backward.

1.0.1 - (2022-04-26)
--------------------


Added
~~~~~

-  Scripts for collecting MNIST.
-  Option to collect grayscale data.


Changed
~~~~~~~

-  Restructure example scripts, i.e. subfolder ``recon`` for reconstructions.
-  Remove heavy installs from setup (e.g. pycsou, lpips, skikit-image).



1.0.0 - (2022-03-21)
--------------------

First version!



