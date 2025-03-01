# #############################################################################
# network_unet.py
# =================
# Adapted from:
# https://github.com/cszn/DPIR/blob/15bca3fcc1f3cc51a1f99ccf027691e278c19354/models/network_unet.py
# #############################################################################


import torch
import torch.nn as nn
import lensless.recon.drunet.basicblock as B
from torchvision.transforms.functional import resize

"""
# ====================
# unet
# From https://github.com/cszn/DPIR/blob/master/main_dpir_denoising.py
# ====================
"""


class UNet(nn.Module):
    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=2,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    ):
        super(UNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], mode="C" + act_mode[-1])

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = B.downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = B.downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError("downsample mode [{:s}] is not found".format(downsample_mode))

        self.m_down1 = B.sequential(
            *[B.conv(nc[0], nc[0], mode="C" + act_mode) for _ in range(nb)],
            downsample_block(nc[0], nc[1], mode="2" + act_mode)
        )
        self.m_down2 = B.sequential(
            *[B.conv(nc[1], nc[1], mode="C" + act_mode) for _ in range(nb)],
            downsample_block(nc[1], nc[2], mode="2" + act_mode)
        )
        self.m_down3 = B.sequential(
            *[B.conv(nc[2], nc[2], mode="C" + act_mode) for _ in range(nb)],
            downsample_block(nc[2], nc[3], mode="2" + act_mode)
        )

        self.m_body = B.sequential(
            *[B.conv(nc[3], nc[3], mode="C" + act_mode) for _ in range(nb + 1)]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = B.upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError("upsample mode [{:s}] is not found".format(upsample_mode))

        self.m_up3 = B.sequential(
            upsample_block(nc[3], nc[2], mode="2" + act_mode),
            *[B.conv(nc[2], nc[2], mode="C" + act_mode) for _ in range(nb)]
        )
        self.m_up2 = B.sequential(
            upsample_block(nc[2], nc[1], mode="2" + act_mode),
            *[B.conv(nc[1], nc[1], mode="C" + act_mode) for _ in range(nb)]
        )
        self.m_up1 = B.sequential(
            upsample_block(nc[1], nc[0], mode="2" + act_mode),
            *[B.conv(nc[0], nc[0], mode="C" + act_mode) for _ in range(nb)]
        )

        self.m_tail = B.conv(nc[0], out_nc, bias=True, mode="C")

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1) + x0

        return x


class UNetRes(nn.Module):
    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        concatenate_compensation=False,
        input_background=False,
        background_subtraction=False,
    ):
        super(UNetRes, self).__init__()

        assert len(nc) == 4, "nc's length should be 4."

        self.nc = nc
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode="C")

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = B.downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = B.downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError("downsample mode [{:s}] is not found".format(downsample_mode))

        self.m_down1 = B.sequential(
            *[B.ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode="2")
        )
        self.m_down2 = B.sequential(
            *[B.ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode="2")
        )
        self.m_down3 = B.sequential(
            *[B.ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode="2")
        )

        # define background subtraction branch (pre-processing)
        self.input_background = input_background
        self.background_subtraction = background_subtraction
        if background_subtraction:
            self.subtraction_weights = nn.Parameter(torch.ones(len(nc)))
            # TODO create downsample block for background
            self.m_head_background = B.conv(in_nc, nc[0], bias=False, mode="C")
            self.m_down1_background = B.sequential(
                *[
                    B.ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C")
                    for _ in range(nb)
                ],
                downsample_block(nc[0], nc[1], bias=False, mode="2")
            )
            self.m_down2_background = B.sequential(
                *[
                    B.ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C")
                    for _ in range(nb)
                ],
                downsample_block(nc[1], nc[2], bias=False, mode="2")
            )
            self.m_down3_background = B.sequential(
                *[
                    B.ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C")
                    for _ in range(nb)
                ],
                downsample_block(nc[2], nc[3], bias=False, mode="2")
            )

        # define compensation branch (post-processing)
        self.concatenate_compensation = concatenate_compensation
        if concatenate_compensation:
            self.m_body = B.sequential(
                B.conv(nc[3] + concatenate_compensation, nc[3], bias=False, mode="C" + act_mode),
                *[
                    B.ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C")
                    for _ in range(nb)
                ]
            )
        else:
            self.m_body = B.sequential(
                *[
                    B.ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C")
                    for _ in range(nb)
                ]
            )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = B.upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError("upsample mode [{:s}] is not found".format(upsample_mode))

        self.m_up3 = B.sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2"),
            *[B.ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C") for _ in range(nb)]
        )
        self.m_up2 = B.sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2"),
            *[B.ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C") for _ in range(nb)]
        )
        self.m_up1 = B.sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2"),
            *[B.ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C") for _ in range(nb)]
        )

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode="C")

    def forward(self, x0, compensation_output=None, background=None):
        if self.concatenate_compensation:
            assert compensation_output is not None, "compensation_output should not be None."
        if self.background_subtraction:
            assert background is not None, "background should not be None."

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        if self.background_subtraction:
            # downsample background
            x1_background = self.m_head_background(background)
            x2_background = self.m_down1_background(x1_background)
            x3_background = self.m_down2_background(x2_background)
            x4_background = self.m_down3_background(x3_background)

            # background subtraction
            x1 = x1 - self.subtraction_weights[0] * x1_background
            x2 = x2 - self.subtraction_weights[1] * x2_background
            x3 = x3 - self.subtraction_weights[2] * x3_background
            x4 = x4 - self.subtraction_weights[3] * x4_background

        if compensation_output is not None:
            compensation_output_re = resize(compensation_output, tuple(x4.shape[-2:]))
            latent = torch.cat([x4, compensation_output_re], dim=1)
        else:
            latent = x4

        x = self.m_body(latent)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        return x
