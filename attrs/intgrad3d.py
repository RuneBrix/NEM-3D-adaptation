import os
import numpy as np
import torch
from captum.attr import IntegratedGradients
from .attribution_template import attribution_template_class


class itg3d_atr(attribution_template_class):
    """
    3D Integrated Gradients.
    """
    def __init__(self, model, train_data=None, use_predicted_labels=False):
        self.attr_name = "intgrad3d"
        self.use_predicted_labels = use_predicted_labels
        self.model = model
        self.n_steps = 64
        self.method = "gausslegendre"
        self._ig = IntegratedGradients(self.model)

        # For preflight/experiments that call gen_mask:
        self.mask_top_ratio = 0.15

        v = os.environ.get("IG_INTERNAL_BATCH_SIZE", "1").strip()
        self.internal_batch_size = None if v.lower() in ("none", "") else int(v)

    def reinit(self, model, train_data=None):
        self.model = model
        self._ig = IntegratedGradients(self.model)

    @torch.no_grad()
    def get_output(self, X):
        return self.model(X)

    @torch.no_grad()
    def _infer_targets(self, X):
        logits = self.model(X).detach()

        # Binary case: single logit per sample
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            B = logits.shape[0] if logits.ndim >= 1 else 1
            return torch.zeros(B, dtype=torch.long, device=X.device)

        # Multiclass: use argmax
        return torch.argmax(logits, dim=1).long().to(X.device)

    def _as_batch5(self, X):
        if X.dim() == 4:
            return X.unsqueeze(0)
        return X

    def _targets_tensor(self, y, X):
        if y is None or self.use_predicted_labels:
            return self._infer_targets(X)
        y = torch.as_tensor(y, dtype=torch.long, device=X.device).view(-1)
        if y.numel() == 1 and X.shape[0] > 1:
            y = y.expand(X.shape[0])
        return y

    def gen_attr(self, X, y=None):
        X = self._as_batch5(X)
        device = X.device
        self.model.eval()

        y_t = self._targets_tensor(y, X)
        baseline = torch.zeros_like(X, device=device)

        X.requires_grad_(True)

        attr_t = self._ig.attribute(
            X,
            baselines=baseline,
            target=y_t,
            n_steps=self.n_steps,
            method=self.method,
            internal_batch_size=self.internal_batch_size,  # <-- key change
        )  # [B,C,D,H,W]

        if attr_t.shape[1] == 1:
            attr_t = attr_t[:, 0]
        else:
            attr_t = attr_t.mean(dim=1)

        attr = attr_t.detach().cpu().numpy().astype(np.float32)
        return attr.squeeze()

    @torch.no_grad()
    def gen_mask(self, X):
        """
        Build a binary mask from IG attributions (top-ratio per sample).
        Returns (mlogits, X*mask, mask).
        """
        Xb = self._as_batch5(X)

        y_pred = self._infer_targets(Xb)
        with torch.enable_grad():
            attr = self.gen_attr(Xb, y_pred)  # np [B,D,H,W]

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
        mlogits = self.get_output(x_masked)
        return mlogits, x_masked, mask


class itg3d_ranked_atr(itg3d_atr):
    def __init__(self, model, train_data=None, use_predicted_labels=False):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "intgrad3d_ranked"

    def gen_attr(self, X, y=None):
        a = super().gen_attr(X, y)
        if a.ndim == 3:
            return self.rank_attr(a)
        return np.stack([self.rank_attr(v) for v in a], axis=0)
