# #############################################################################
# trainable_mask.py
# ==================
# Authors :
# Yohann PERRON [yohann.perron@gmail.com]
# Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import abc
import omegaconf
import numpy as np
import torch
from lensless.utils.image import is_grayscale, rgb2gray
from lensless.hardware.slm import full2subpattern, get_programmable_mask, get_intensity_psf
from lensless.hardware.sensor import VirtualSensor
from waveprop.devices import slm_dict
from lensless.hardware.mask import CodedAperture


class TrainableMask(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract class for defining trainable masks.

    The following abstract methods need to be defined:

    - :py:class:`~lensless.hardware.trainable_mask.TrainableMask.get_psf`: returning the PSF of the mask.
    - :py:class:`~lensless.hardware.trainable_mask.TrainableMask.project`: projecting the mask parameters to a valid space (should be a subspace of [0,1]).

    """

    def __init__(self, optimizer="Adam", lr=1e-3, **kwargs):
        """
        Base constructor. Derived constructor may define new state variables

        Parameters
        ----------
        optimizer : str, optional
            Optimizer to use for updating the mask parameters, by default "Adam"
        lr : float, optional
            Learning rate for the mask parameters, by default 1e-3
        """
        super().__init__()
        self._optimizer = optimizer
        self._lr = lr
        self._counter = 0

    def _set_optimizer(self, param):
        """Set the optimizer for the mask parameters."""
        self._optimizer = getattr(torch.optim, self._optimizer)(param, lr=self._lr)

    @abc.abstractmethod
    def get_psf(self):
        """
        Abstract method for getting the PSF of the mask. Should be fully compatible with pytorch autograd.

        Returns
        -------
        :py:class:`~torch.Tensor`
            The PSF of the mask.
        """
        raise NotImplementedError

    def update_mask(self):
        """Update the mask parameters. According to externaly updated gradiants."""
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self.project()
        self._counter += 1

    @abc.abstractmethod
    def project(self):
        """Abstract method for projecting the mask parameters to a valid space (should be a subspace of [0,1])."""
        raise NotImplementedError


class TrainablePSF(TrainableMask):
    # class TrainablePSF(torch.nn.Module, TrainableMask):
    """
    Class for defining an object that directly optimizes the PSF, without any constraints on what can be realized physically.

    Parameters
    ----------
    grayscale : bool, optional
        Whether mask should be returned as grayscale when calling :py:class:`~lensless.hardware.trainable_mask.TrainableMask.get_psf`.
        Otherwise PSF will be returned as RGB. By default False.
    """

    def __init__(self, initial_psf, grayscale=False, **kwargs):
        super().__init__(**kwargs)
        self._psf = torch.nn.Parameter(initial_psf)
        initial_param = [self._psf]
        self._set_optimizer(initial_param)

        # checks
        assert len(initial_psf.shape) == 4, "Mask must be of shape (depth, height, width, channels)"
        self.grayscale = grayscale
        self._is_grayscale = is_grayscale(initial_psf)
        if grayscale:
            assert self._is_grayscale, "PSF must be grayscale"

    def get_psf(self):
        if self._is_grayscale:
            if self.grayscale:
                # simulation in grayscale
                return self._psf
            else:
                # replicate to 3 channels
                return self._psf.expand(-1, -1, -1, 3)
        else:
            # assume RGB
            return self._psf

    def project(self):
        self._psf.data = torch.clamp(self._psf, 0, 1)


class AdafruitLCD(TrainableMask):
    def __init__(
        self,
        initial_vals,
        sensor,
        slm,
        train_mask_vals=True,
        color_filter=None,
        rotate=None,
        flipud=False,
        use_waveprop=False,
        vertical_shift=None,
        horizontal_shift=None,
        scene2mask=None,
        mask2sensor=None,
        deadspace=True,
        downsample=None,
        min_val=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        initial_vals : :py:class:`~torch.Tensor`
            Initial mask parameters.
        sensor : :py:class:`~lensless.hardware.sensor.VirtualSensor`
            Sensor object.
        slm_param : :py:class:`~lensless.hardware.slm.SLMParam`
            SLM parameters.
        rotate : float, optional
            Rotation angle in degrees, by default None.
        flipud : bool, optional
            Whether to flip the mask vertically, by default False.
        use_waveprop : bool, optional
            Whether to use wave propagation for simulating PSF. If False, PSF will simply be intensity of mask pattern, by default False.
        vertical_shift : int, optional
            Vertical shift of the mask, by default None.
        horizontal_shift : int, optional
            Horizontal shift of the mask, by default None.
        scene2mask : :py:class:`~torch.Tensor`, optional
            Distance from scene to mask. Used for wave propagation, by default None.
        mask2sensor : :py:class:`~torch.Tensor`, optional
            Distance from mask to sensor. Used for wave propagation, by default None.
        downsample : int, optional
            Downsample factor, by default None.
        min_val : float, optional
            Minimum value for mask weights, by default 0.
        """

        super().__init__(**kwargs)

        self.train_mask_vals = train_mask_vals
        if train_mask_vals:
            self._vals = torch.nn.Parameter(initial_vals)
        else:
            self._vals = initial_vals

        initial_param = None
        if color_filter is not None:
            self._color_filter = torch.nn.Parameter(color_filter)
            if train_mask_vals:
                initial_param = [self._vals, self._color_filter]
            else:
                initial_param = [self._color_filter]
        else:
            assert (
                train_mask_vals
            ), "If color filter is not trainable, mask values must be trainable"
            initial_param = [self._vals]
            self._color_filter = None
        assert initial_param is not None, "Initial parameters must be set"

        # set optimizer
        self._set_optimizer(initial_param)

        self.slm_param = slm_dict[slm]
        self.device = slm
        self.sensor = VirtualSensor.from_name(sensor, downsample=downsample)
        self.rotate = rotate
        self.flipud = flipud
        self.use_waveprop = use_waveprop
        self.scene2mask = scene2mask
        self.mask2sensor = mask2sensor
        self.deadspace = deadspace
        self.vertical_shift = vertical_shift
        self.horizontal_shift = horizontal_shift
        self.min_val = min_val
        if downsample is not None and vertical_shift is not None:
            self.vertical_shift = vertical_shift // downsample
        if downsample is not None and horizontal_shift is not None:
            self.horizontal_shift = horizontal_shift // downsample
        if self.use_waveprop:
            assert self.scene2mask is not None
            assert self.mask2sensor is not None

    def get_psf(self):

        mask = get_programmable_mask(
            vals=self._vals,
            sensor=self.sensor,
            slm_param=self.slm_param,
            rotate=self.rotate,
            flipud=self.flipud,
            color_filter=self._color_filter,
            deadspace=self.deadspace,
        )

        if self.vertical_shift is not None:
            mask = torch.roll(mask, self.vertical_shift, dims=1)

        if self.horizontal_shift is not None:
            mask = torch.roll(mask, self.horizontal_shift, dims=2)

        psf_in = get_intensity_psf(
            mask=mask,
            sensor=self.sensor,
            waveprop=self.use_waveprop,
            scene2mask=self.scene2mask,
            mask2sensor=self.mask2sensor,
        )

        # add first dimension (depth)
        psf_in = psf_in.unsqueeze(0)

        # move channels to last dimension
        psf_in = psf_in.permute(0, 2, 3, 1)

        # flip mask
        psf_in = torch.flip(psf_in, dims=[-3, -2])

        # normalize
        psf_in = psf_in / psf_in.norm()

        return psf_in

    def project(self):
        if self.train_mask_vals:
            self._vals.data = torch.clamp(self._vals, self.min_val, 1)
        if self._color_filter is not None:
            self._color_filter.data = torch.clamp(self._color_filter, 0, 1)
            # normalize each row to 1
            self._color_filter.data = self._color_filter / self._color_filter.sum(
                dim=[1, 2]
            ).unsqueeze(-1).unsqueeze(-1)


class TrainableCodedAperture(TrainableMask):
    def __init__(
        self,
        sensor_name,
        downsample=None,
        binary=True,
        torch_device="cuda",
        **kwargs,
    ):
        """
        TODO: Distinguish between separable and non-separable.
        """

        # 1) call base constructor so parameters can be set
        super().__init__(**kwargs)

        # 2) initialize mask
        assert "distance_sensor" in kwargs, "Distance to sensor must be specified"
        assert "method" in kwargs, "Method must be specified."
        assert "n_bits" in kwargs, "Number of bits must be specified."
        self._mask_obj = CodedAperture.from_sensor(
            sensor_name,
            downsample,
            is_torch=True,
            torch_device=torch_device,
            **kwargs,
        )

        # 3) set learnable parameters (should be immediate attributes of the class)
        self._row = None
        self._col = None
        self._mask = None
        if self._mask_obj.row is not None:
            # separable
            self.separable = True
            self._row = torch.nn.Parameter(self._mask_obj.row)
            self._col = torch.nn.Parameter(self._mask_obj.col)
            initial_param = [self._row, self._col]
        else:
            # non-separable
            self.separable = False
            self._mask = torch.nn.Parameter(self._mask_obj.mask)
            initial_param = [self._mask]
        self.binary = binary

        # 4) set optimizer
        self._set_optimizer(initial_param)

        # 5) compute PSF
        self._psf = None
        self.project()

    def get_psf(self):
        return self._psf

    def project(self):
        with torch.no_grad():
            if self.separable:
                self._row.data = torch.clamp(self._row, 0, 1)
                self._col.data = torch.clamp(self._col, 0, 1)
                if self.binary:
                    self._row.data = torch.round(self._row)
                    self._col.data = torch.round(self._col)
            else:
                self._mask.data = torch.clamp(self._mask, 0, 1)
                if self.binary:
                    self._mask.data = torch.round(self._mask)

        # recompute PSF
        self._mask_obj.create_mask(self._row, self._col, mask=self._mask)
        self._mask_obj.compute_psf()
        self._psf = self._mask_obj.psf.unsqueeze(0)
        self._psf = self._psf / self._psf.norm()


"""
Utility functions to help prepare trainable masks.
"""

mask_type_to_class = {
    "AdafruitLCD": AdafruitLCD,
    "TrainablePSF": TrainablePSF,
    "TrainableCodedAperture": TrainableCodedAperture,
    "TrainableHeightVarying": None,
    "TrainableMultiLensArray": None,
}


def prep_trainable_mask(config, psf=None, downsample=None):

    mask = None
    color_filter = None
    downsample = config["files"]["downsample"] if downsample is None else downsample
    mask_type = config["trainable_mask"]["mask_type"]
    if mask_type is not None:
        assert mask_type in mask_type_to_class.keys(), (
            f"Trainable mask type {mask_type} not supported. "
            f"Supported types are {mask_type_to_class.keys()}"
        )
        mask_class = mask_type_to_class[mask_type]

        # -- trainable mask object
        if isinstance(config["trainable_mask"]["initial_value"], omegaconf.dictconfig.DictConfig):

            # from mask config
            mask = mask_class(
                # mask = TrainableCodedAperture(
                sensor_name=config.simulation.sensor,
                downsample=downsample,
                distance_sensor=config.simulation.mask2sensor,
                optimizer=config.trainable_mask.optimizer,
                lr=config.trainable_mask.mask_lr,
                binary=config.trainable_mask.binary,
                torch_device=config.torch_device,
                **config.trainable_mask.initial_value,
            )

        else:

            if config["trainable_mask"]["initial_value"] == "random":
                if psf is not None:
                    initial_mask = torch.rand_like(psf)
                else:
                    sensor = VirtualSensor.from_name(
                        config["simulation"]["sensor"], downsample=downsample
                    )
                    resolution = sensor.resolution
                    initial_mask = torch.rand((1, *resolution, 3))

            elif config["trainable_mask"]["initial_value"] == "psf":

                if config["reconstruction"]["method"] == "svdeconvnet":
                    # learnable PSF as part of SVDeconvNet
                    n_psf = config["reconstruction"]["svdeconvnet"]["K"] ** 2
                    initial_mask = psf.repeat(n_psf, 1, 1, 1)
                else:
                    initial_mask = psf.clone()

            # if file ending with "npy"
            elif config["trainable_mask"]["initial_value"].endswith("npy"):

                pattern = np.load(config["trainable_mask"]["initial_value"])

                initial_mask = full2subpattern(
                    pattern=pattern,
                    shape=config["trainable_mask"]["ap_shape"],
                    center=config["trainable_mask"]["ap_center"],
                    slm=config["trainable_mask"]["slm"],
                )
                initial_mask = torch.from_numpy(initial_mask.astype(np.float32))

                # prepare color filter if needed
                from waveprop.devices import slm_dict
                from waveprop.devices import SLMParam as SLMParam_wp

                slm_param = slm_dict[config["trainable_mask"]["slm"]]
                if (
                    config["trainable_mask"]["train_color_filter"]
                    and SLMParam_wp.COLOR_FILTER in slm_param.keys()
                ):
                    color_filter = slm_param[SLMParam_wp.COLOR_FILTER]
                    color_filter = torch.from_numpy(color_filter.copy()).to(dtype=torch.float32)

                    # TODO: add small random values?
                    color_filter = color_filter + 0.1 * torch.rand_like(color_filter)

            else:
                raise ValueError(
                    f"Initial PSF value {config['trainable_mask']['initial_value']} not supported"
                )

            # convert to grayscale if needed
            if config["trainable_mask"]["grayscale"] and not is_grayscale(initial_mask):
                initial_mask = rgb2gray(initial_mask)

            mask = mask_class(
                initial_mask,
                downsample=downsample,
                color_filter=color_filter,
                **config["trainable_mask"],
            )

    return mask
