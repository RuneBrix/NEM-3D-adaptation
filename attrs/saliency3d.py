import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import Saliency

from .attribution_template import attribution_template_class


def select_sigma(model):
    if model.__class__.__name__ == "ResNet":
        return 2
    elif model.__class__.__name__ == "VGG":
        return 5
    elif model.__class__.__name__ == "ConvNeXt":
        return 3
    elif model.__class__.__name__ == "VisionTransformer":
        return 12
    else:
        return 2


def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0, device=None, dtype=None) -> torch.Tensor:
    if sigma is None or sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)

    radius = int(truncate * float(sigma) + 0.5)
    if radius <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)

    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * float(sigma) ** 2))
    k = k / (k.sum() + 1e-12)
    return k


def gaussian_smooth_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x: [B, 1, D, H, W] (or [B, C, D, H, W] but intended C=1 for attribution maps)
    returns same shape as x
    """
    if sigma is None or sigma <= 0:
        return x

    device = x.device
    dtype = x.dtype

    kz = _gaussian_kernel_1d(sigma, device=device, dtype=dtype)
    ky = _gaussian_kernel_1d(sigma, device=device, dtype=dtype)
    kx = _gaussian_kernel_1d(sigma, device=device, dtype=dtype)

    # Create separable 3D kernel via outer products
    k3 = kz[:, None, None] * ky[None, :, None] * kx[None, None, :]
    k3 = k3[None, None, ...]  # [1,1,kD,kH,kW]

    pad_d = k3.shape[-3] // 2
    pad_h = k3.shape[-2] // 2
    pad_w = k3.shape[-1] // 2

    return F.conv3d(x, k3, padding=(pad_d, pad_h, pad_w))


class saliency3d_atr(attribution_template_class):
    """
    3D Saliency (Captum):
      - Handles inputs shaped (D,H,W), (C,D,H,W), (B,C,D,H,W)
      - Handles binary (single logit) and multiclass (K logits)
      - Returns numpy attribution maps:
          - [D,H,W] for a single sample
          - [B,D,H,W] for a batch
    """
    def __init__(self, model, train_data=None, use_predicted_labels=False):
        self.attr_name = "saliency3d"
        self.use_predicted_labels = use_predicted_labels
        self.model = model
        self._sal = Saliency(self.model)
        self.mask_top_ratio = 0.15

    def reinit(self, model, train_data=None):
        self.model = model
        self._sal = Saliency(self.model)

    def _as_batch5(self, X: torch.Tensor) -> torch.Tensor:
        # Accept:
        #   (D,H,W) -> (1,1,D,H,W)
        #   (C,D,H,W) -> (1,C,D,H,W)
        #   (B,C,D,H,W) -> unchanged
        if X.dim() == 3:
            return X.unsqueeze(0).unsqueeze(0)
        if X.dim() == 4:
            return X.unsqueeze(0)
        if X.dim() == 5:
            return X
        raise ValueError(f"saliency3d_atr: expected 3D/4D/5D tensor, got shape {tuple(X.shape)}")

    @torch.no_grad()
    def _infer_targets(self, Xb: torch.Tensor):
        """
        For multiclass: argmax class index per sample.
        For binary (one logit): target is not meaningful; handled in gen_attr.
        """
        logits = self.model(Xb)
        if logits.ndim == 2 and logits.shape[1] > 1:
            return torch.argmax(logits, dim=1).long().to(Xb.device)
        return None

    def _targets_tensor(self, y, Xb: torch.Tensor):
        if y is None or self.use_predicted_labels:
            return self._infer_targets(Xb)
        y = torch.as_tensor(y, dtype=torch.long, device=Xb.device).view(-1)
        if y.numel() == 1 and Xb.shape[0] > 1:
            y = y.expand(Xb.shape[0])
        return y

    def gen_attr(self, X, y=None):
        Xb = self._as_batch5(X)
        self.model.eval()

        # Decide binary vs multiclass from output shape
        with torch.no_grad():
            logits = self.model(Xb)
            is_multiclass = (logits.ndim == 2 and logits.shape[1] > 1)
            is_binary_2d = (logits.ndim == 2 and logits.shape[1] == 1)
            is_binary_1d = (logits.ndim == 1)

        if is_multiclass:
            target = self._targets_tensor(y, Xb)  # tensor [B]
        else:
            # Binary / single-logit:
            target = 0 if is_binary_2d else None

        # Need grads for saliency
        Xb = Xb.clone().detach().requires_grad_(True)
        try:
            self.model.zero_grad(set_to_none=True)
        except TypeError:
            self.model.zero_grad()

        attr_t = self._sal.attribute(Xb, target=target)  # [B,C,D,H,W]

        # collapse channels -> [B,D,H,W]
        if attr_t.shape[1] == 1:
            attr_t = attr_t[:, 0]
        else:
            attr_t = attr_t.mean(dim=1)

        attr = attr_t.detach().cpu().numpy().astype(np.float32)
        return attr.squeeze()

    @torch.no_grad()
    def gen_mask(self, X):
        """
        Optional helper like in itg3d_atr: create a top-ratio binary mask from saliency.
        Returns (mlogits, X*mask, mask).
        """
        Xb = self._as_batch5(X)
        y_pred = self._infer_targets(Xb)
        attr = self.gen_attr(Xb, y_pred)  # np [B,D,H,W] or [D,H,W]
        if attr.ndim == 3:
            attr = attr[None, ...]

        B, D, H, W = attr.shape
        attr_t = torch.from_numpy(attr).to(device=Xb.device, dtype=torch.float32).view(B, 1, D, H, W)

        masks = []
        for b in range(B):
            flat = attr_t[b].flatten()
            k = max(1, int(self.mask_top_ratio * flat.numel()))
            if k >= flat.numel():
                thr = flat.min()
            else:
                thr = torch.topk(flat, k).values.min()
            masks.append((attr_t[b] >= thr).float())

        mask = torch.stack(masks, dim=0)  # [B,1,D,H,W]
        x_masked = Xb * mask
        mlogits = self.model(x_masked)
        return mlogits, x_masked, mask


class saliency3d_forgrad_atr(saliency3d_atr):
    def __init__(self, model, train_data=None, use_predicted_labels=False):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "saliency3d_forgrad"
        self.sigma = select_sigma(model)

    def reinit(self, model, train_data=None):
        super().reinit(model, train_data)
        self.sigma = select_sigma(model)

    def gen_attr(self, X, y=None):
        # Get raw saliency first
        a = super().gen_attr(X, y)  # np [D,H,W] or [B,D,H,W]

        # Smooth in torch with conv3d
        if a.ndim == 3:
            a_t = torch.from_numpy(a).to(self.model.parameters().__next__().device).float()[None, None, ...]
            a_s = gaussian_smooth_3d(a_t, sigma=float(self.sigma))[0, 0]
            return a_s.detach().cpu().numpy().astype(np.float32)

        if a.ndim == 4:
            a_t = torch.from_numpy(a).to(self.model.parameters().__next__().device).float()[:, None, ...]
            a_s = gaussian_smooth_3d(a_t, sigma=float(self.sigma))[:, 0]
            return a_s.detach().cpu().numpy().astype(np.float32)

        raise ValueError(f"saliency3d_forgrad_atr: unexpected attribution shape {a.shape}")
