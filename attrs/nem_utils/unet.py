from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SeparableConv2dBnAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        act_layer,
        norm_layer,
        padding=0,
        stride=1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
            padding_mode="reflect",
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DepthwiseSeparableConvTranspose(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=2,
        kernel_size=4,
        padding=1,
        act_layer=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=scale_factor,
            padding=padding,
            output_padding=0,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(
        self,
        x_channels,  # Only the channels of the input to upsample
        skip_channels,  # Channels from skip connection (if any)
        out_channels,
        scale_factor=2,
        act_layer=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, act_layer=act_layer)

        if scale_factor != 1:
            self.up = nn.Sequential(
                # nn.Upsample(scale_factor=scale_factor, mode="nearest"),
                DepthwiseSeparableConvTranspose(
                    x_channels,
                    x_channels,
                    scale_factor=scale_factor,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ),
                SeparableConv2dBnAct(
                    x_channels,
                    x_channels,
                    kernel_size=3,
                    padding=1,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ),
            )
        else:
            self.up = nn.Identity()

        # Total input channels = upsampled + skip
        in_channels = x_channels + skip_channels

        self.conv1 = SeparableConv2dBnAct(
            in_channels, out_channels, norm_layer=norm_layer, **conv_args
        )
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv2 = SeparableConv2dBnAct(
            out_channels, out_channels, norm_layer=norm_layer, **conv_args
        )

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        final_channels,
        norm_layer,
        center,
        use_img_space,
    ):
        super().__init__()

        self.center = nn.Identity()
        if center:
            self.center = DecoderBlock(
                encoder_channels[0],
                encoder_channels[0],
                scale_factor=1.0,
                norm_layer=norm_layer,
            )

        skip_channels = (
            list(encoder_channels[1:]) + [0]
            if not use_img_space
            else encoder_channels[1:]
        )
        upsample_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        in_channels = np.array(upsample_channels) + np.array(skip_channels)
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        for x_chs, skip_chs, out_chs in zip(
            upsample_channels, skip_channels, out_channels
        ):
            self.blocks.append(
                DecoderBlock(
                    x_channels=x_chs,
                    skip_channels=skip_chs,
                    out_channels=out_chs,
                    norm_layer=norm_layer,
                )
            )

        self.final_conv = nn.Conv2d(
            out_channels[-1], final_channels, kernel_size=(1, 1)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        if x.shape[2] != 224:
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = self.final_conv(x)
        return x

class Unet(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        freeze_backbone=False,
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),
        num_classes=1,
        center=False,
        norm_layer=nn.BatchNorm2d,
        encoder_channels=None,
        use_img_space=False,
    ):
        super().__init__()

        encoder_channels = encoder_channels[::-1]
        self.encoder = backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if not decoder_use_batchnorm:
            norm_layer = None

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
            use_img_space=use_img_space,
        )

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x.reverse()
        x = self.decoder(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv3dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act_layer=nn.SiLU, norm_layer=nn.BatchNorm3d, padding=0, stride=1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparableConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=4, padding=1,
                 act_layer=nn.SiLU, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.depthwise = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=scale_factor, padding=padding, output_padding=0, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DecoderBlock3D(nn.Module):
    def __init__(self, x_channels, skip_channels, out_channels, scale_factor=2,
                 act_layer=nn.SiLU, norm_layer=nn.BatchNorm3d):
        super().__init__()
        if scale_factor != 1:
            self.up = nn.Sequential(
                DepthwiseSeparableConvTranspose3d(
                    x_channels, x_channels, scale_factor=scale_factor,
                    act_layer=act_layer, norm_layer=norm_layer
                ),
                SeparableConv3dBnAct(
                    x_channels, x_channels, kernel_size=3, padding=1,
                    act_layer=act_layer, norm_layer=norm_layer
                ),
            )
        else:
            self.up = nn.Identity()

        in_channels = x_channels + skip_channels
        self.conv1 = SeparableConv3dBnAct(in_channels, out_channels, kernel_size=3, padding=1,
                                          act_layer=act_layer, norm_layer=norm_layer)
        self.dropout = nn.Dropout3d(p=0.2)
        self.conv2 = SeparableConv3dBnAct(out_channels, out_channels, kernel_size=3, padding=1,
                                          act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x, skip: torch.Tensor | None = None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class UnetDecoder3D(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, final_channels, norm_layer=nn.BatchNorm3d, center=False):
        super().__init__()
        self.center = nn.Identity()
        if center:
            self.center = DecoderBlock3D(encoder_channels[0], encoder_channels[0],
                                         out_channels=encoder_channels[0], scale_factor=1.0,
                                         norm_layer=norm_layer)

        skip_channels = list(encoder_channels[1:]) + [0]
        upsample_channels = [encoder_channels[0]] + list(decoder_channels[:-1])

        self.blocks = nn.ModuleList()
        for x_chs, skip_chs, out_chs in zip(upsample_channels, skip_channels, decoder_channels):
            self.blocks.append(
                DecoderBlock3D(x_channels=x_chs, skip_channels=skip_chs, out_channels=out_chs,
                               norm_layer=norm_layer)
            )

        self.final_conv = nn.Conv3d(decoder_channels[-1], final_channels, kernel_size=1)

        # init
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats: list[torch.Tensor]):
        x = feats[0]
        skips = feats[1:]
        x = self.center(x)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        return self.final_conv(x)


class Unet3D(nn.Module):
    """
    3D counterpart to the existing 2D Unet class. Expects encoder(backbone) to return
    a list of feature maps highest->lowest resolution.
    """
    def __init__(self, backbone: nn.Module, freeze_backbone=False,
                 decoder_channels=(256, 128, 64, 32), num_classes=1,
                 center=False, norm_layer=nn.BatchNorm3d, encoder_channels=None):
        super().__init__()
        assert encoder_channels is not None, "Unet3D requires encoder_channels"
        encoder_channels = encoder_channels[::-1]  # match top-down order
        self.encoder = backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = UnetDecoder3D(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
        )

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)  # list high->low
        feats.reverse()
        x = self.decoder(feats)
        return x
