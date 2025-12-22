import torch
import torch.nn as nn
import timm
from torch.nn.functional import interpolate
import torch.nn.functional as F

class resnet50_img_extractor(torch.nn.Module):
    def __init__(
        self,
        model=None,
    ):
        super().__init__()
        if model is None:
            model = timm.create_model("resnet50", pretrained=True)
        self.core_model = model

        self.channels = (3, 64, 256, 512, 1024, 2048)
        self.shapes = (
            112,
            56,
            28,
            14,
            7,
        )
        self.channels = (64, 256, 512, 1024, 2048)

        self.output_shape = (-1, 2048, 7, 7)
        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        embs = []
        x = self.core_model.conv1(x)
        x = self.core_model.bn1(x)
        x = self.core_model.act1(x)
        embs += [x]
        x = self.core_model.maxpool(x)

        x = self.core_model.layer1(x)
        embs += [x]
        x = self.core_model.layer2(x)
        embs += [x]
        x = self.core_model.layer3(x)
        embs += [x]
        x = self.core_model.layer4(x)
        embs += [x]
        return embs

    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)

    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)


class convnext_extractor(torch.nn.Module):
    def __init__(
        self,
        model=None,
    ):
        super().__init__()
        if model is None:
            model = timm.create_model("convnext_small", pretrained=True)
        self.core_model = model
        self.shapes = (
            56,
            56,
            28,
            14,
            7,
        )
        self.channels = (96, 96, 192, 384, 768)
        self.output_shape = (-1, 768, 7, 7)
        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        embs = []
        x = self.core_model.stem(x)
        embs.append(x)
        for stage in self.core_model.stages[:-1]:
            x = stage(x)
            embs.append(x)
        x = self.core_model.stages[-1](x)
        x = self.core_model.norm_pre(x)
        embs.append(x)
        return embs

    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)

    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)


class vgg_img_extractor(torch.nn.Module):
    def __init__(
        self,
        model=None,
    ):
        super().__init__()
        if model is None:
            model = timm.create_model("vgg16", pretrained=True)
        self.core_model = model
        self.shapes = (
            112,
            56,
            28,
            14,
            7,
        )
        self.channels = (64, 128, 256, 512, 512)
        self.output_shape = (-1, 512, 7, 7)
        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        embs = []
        for i, layer in enumerate(self.core_model.features):
            x = layer(x)
            if i in [4, 9, 16, 23, 30]:
                embs.append(x)
        return embs

    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)

    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)


class vit_feature_extractor(torch.nn.Module):
    def __init__(
        self,
        core_model=None,
        shapes=(
            112,
            56,
            28,
            14,
            7,
        ),
        # channels=(12, 48, 192, 768, 3072),
        channels=(768, 768, 768, 768, 768),
        output_shape=(-1, 768, 14, 14),
    ):
        super().__init__()
        if core_model is None:
            core_model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.core_model = core_model
        self.shapes = shapes
        self.channels = channels
        self.output_shape = output_shape
        self.pretrained_cfg = self.core_model.pretrained_cfg
        self.selected_layers = [2, 4, 6, 8, 10, 12]
        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        out = self.core_model.get_intermediate_layers(
            x, self.selected_layers, reshape=True
        )
        return [
            interpolate(rep, size=shap, mode="nearest")
            for rep, shap in zip(out, self.shapes)
        ]

    # return [
    #     rep.reshape((-1, channel, shap, shap))
    #     for rep, shap, channel in zip(out, self.shapes, self.channels)
    # ]

    def get_embeddings(self, x: torch.Tensor):
        return self.core_model.forward_features(x)

    def get_output_from_embeddings(self, features: torch.Tensor):
        return self.core_model.forward_head(features)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)
    
class DenseNet3DExtractor(nn.Module):
    """
    Wrap a MONAI 3D DenseNet (e.g., DenseNet121 with spatial_dims=3)
    to expose multi-scale features + logits, matching the 2D extractor API.
    """
    def __init__(self, model=None):
        super().__init__()
        if model is None:
            from monai.networks.nets import DenseNet121
            model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
        self.core_model = model

        # Break out the MONAI DenseNet feature stem/blocks so we can collect maps
        feats = self.core_model.features
        self.stem = nn.Sequential(feats.conv0, feats.norm0, feats.relu0, feats.pool0)

        self.db1 = feats.denseblock1; self.tr1 = feats.transition1
        self.db2 = feats.denseblock2; self.tr2 = feats.transition2
        self.db3 = feats.denseblock3; self.tr3 = feats.transition3
        self.db4 = feats.denseblock4; self.norm5 = feats.norm5

        # Channel counts per collected stage
        self.channels = (1024, 512, 256, 128)

        # Keep parity with 2D extractors
        self.use_img_space = False

    def forward(self, x):
        m = self.core_model

        # Stem
        x = m.features.conv0(x)
        x = m.features.norm0(x)
        x = m.features.relu0(x)
        x = m.features.pool0(x)

        # Block 1 -> Transition 1
        x = m.features.denseblock1(x)
        x = m.features.transition1(x)
        t1 = x              # 128 ch

        # Block 2 -> Transition 2
        x = m.features.denseblock2(x)
        x = m.features.transition2(x)
        t2 = x              # 256 ch

        # Block 3 -> Transition 3
        x = m.features.denseblock3(x)
        x = m.features.transition3(x)
        t3 = x              # 512 ch

        # Block 4 (no transition)
        x = m.features.denseblock4(x)
        x = m.features.norm5(x)
        head = x            # 1024 ch

        # return high->low resolution list (must match self.channels)
        return [head, t3, t2, t1]


    def get_embeddings(self, x):
        feats = self.forward(x)[0]                  # head
        pooled = F.adaptive_avg_pool3d(feats, 1).flatten(1)
        return pooled

    def get_output_from_embeddings(self, features):
        # DenseNet121 final classifier is in core_model.class_layers.out
        return self.core_model.class_layers.out(features)

    def get_output(self, x):
        return self.core_model(x)



class UNetEnc3DExtractor(nn.Module):
    """
    Wrap UNetEncoderClassifier (3D) to expose hierarchical feature maps + logits,
    matching the same interface used by the 2D extractors.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.core_model = model

        # Encoder blocks from training code
        self.enc1 = model.enc1; self.pool1 = model.pool1
        self.enc2 = model.enc2; self.pool2 = model.pool2
        self.enc3 = model.enc3; self.pool3 = model.pool3
        self.enc4 = model.enc4
        self.head = model.head

        # Channel sizes of encoder stages; update if trained with other widths
        self.channels = (32, 64, 128, 256)

        self.use_img_space = False

    def forward(self, x: torch.Tensor):
        """Return list of feature maps from high-res -> low-res."""
        embs = []
        x = self.enc1(x); embs.append(x)
        x = self.pool1(x)

        x = self.enc2(x); embs.append(x)
        x = self.pool2(x)

        x = self.enc3(x); embs.append(x)
        x = self.pool3(x)

        x = self.enc4(x); embs.append(x)
        return embs

    def get_embeddings(self, x: torch.Tensor):
        """Penultimate encoder output (before global pool + linear head)."""
        x = self.enc1(x); x = self.pool1(x)
        x = self.enc2(x); x = self.pool2(x)
        x = self.enc3(x); x = self.pool3(x)
        x = self.enc4(x)
        return x

    def get_output_from_embeddings(self, features: torch.Tensor):
        x = F.adaptive_avg_pool3d(features, 1).flatten(1)
        return self.head(x)  # (B, 1)

    def get_output(self, x: torch.Tensor):
        return self.core_model(x)