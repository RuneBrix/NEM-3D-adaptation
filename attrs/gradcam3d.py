import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attribution_template import attribution_template_class


class gradcam3d_atr(attribution_template_class):
    """
    3D Grad-CAM / Grad-CAM++ for volumetric classifiers.

    Key improvements vs a naive 3D Grad-CAM:
      - Default target for MONAI DenseNet is *not* model.features (too low-res),
        but model.features.denseblock3 (higher spatial resolution).
      - Supports Grad-CAM++ (often sharper + less "washed out").
      - Supports multi-layer CAM fusion (mean/sum/max).

    Input X:
      - [B, C, D, H, W] or [C, D, H, W]
    Output:
      - [D, H, W] or [B, D, H, W]
    """

    def __init__(
        self,
        model: nn.Module,
        train_data=None,
        use_predicted_labels: bool = False,
        *,
        method: str = "gradcam++",
        target_layer_name: str | None = None,
        target_layer_names: list[str] | None = None,
        fuse_layers: str = "mean",                 # "mean" | "sum" | "max"
        apply_relu: bool = True,
        smooth_kernel: int = 0,
        normalise_for_mask: bool = True,
        min_target_spatial: tuple[int, int, int] = (4, 4, 4),
    ):
        self.attr_name = "gradcam3d"
        self.model = model
        self.use_predicted_labels = use_predicted_labels

        # For deletion/insertion-style experiments
        self.mask_top_ratio = 0.15

        # CAM options
        self.method = method.lower().strip()
        if self.method not in ("gradcam", "gradcam++"):
            raise ValueError(f"Unknown method='{method}'. Use 'gradcam' or 'gradcam++'.")

        self.fuse_layers = fuse_layers.lower().strip()
        if self.fuse_layers not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown fuse_layers='{fuse_layers}'. Use 'mean'|'sum'|'max'.")

        self.apply_relu = bool(apply_relu)
        self.smooth_kernel = int(smooth_kernel) if smooth_kernel is not None else 0
        if self.smooth_kernel and (self.smooth_kernel < 3 or self.smooth_kernel % 2 == 0):
            raise ValueError("smooth_kernel must be 0 or an odd int >= 3")

        self.normalise_for_mask = bool(normalise_for_mask)
        self.min_target_spatial = tuple(int(x) for x in min_target_spatial)

        # Grad-CAM internals: can be single layer or multiple layers
        self._handles = []
        self._target_layers: list[tuple[str, nn.Module]] = []
        self.activations: dict[str, torch.Tensor] = {}

        # Resolve target layer(s)
        if target_layer_names is not None and len(target_layer_names) > 0:
            self._target_layers = [(n, self._find_module_by_name(model, n)) for n in target_layer_names]
        elif target_layer_name is not None:
            self._target_layers = [(target_layer_name, self._find_module_by_name(model, target_layer_name))]
        else:
            # Auto-select sensible defaults
            self._target_layers = self._auto_select_target_layers(model)

        if len(self._target_layers) == 0:
            raise ValueError("gradcam3d_atr: no target layers found.")

        self._register_hooks()

    def _find_module_by_name(self, model: nn.Module, name: str) -> nn.Module:
        """
        Find an nn.Module by exact match in model.named_modules().
        """
        for n, m in model.named_modules():
            if n == name:
                return m
        raise ValueError(f"Could not find module named '{name}' in model.named_modules().")

    def _auto_select_target_layers(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """
        Heuristic defaults:
          - If MONAI DenseNet-style: prefer features.denseblock3 (higher res),
            optionally also include denseblock4 for fusion.
          - Otherwise: fall back to last Conv3d (single layer).
        """
        named = dict(model.named_modules())

        if "features.denseblock3" in named:
            layers = [("features.denseblock3", named["features.denseblock3"])]
            if "features.denseblock4" in named:
                layers.append(("features.denseblock4", named["features.denseblock4"]))
            return layers
        
        for cand in ("features.denseblock4", "features.transition3", "features.denseblock2", "features"):
            if cand in named:
                return [(cand, named[cand])]

        # Generic fallback: last Conv3d in the whole model
        last_name, last_conv = None, None
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv3d):
                last_name, last_conv = n, m
        if last_conv is None:
            raise ValueError("gradcam3d_atr: could not find a suitable layer (no Conv3d in model).")
        return [(last_name, last_conv)]

    def _register_hooks(self):
        """
        Attach forward hooks to capture activations per target layer.
        """
        self._remove_hooks()
        self.activations = {}

        def make_forward_hook(layer_name: str):
            def forward_hook(module, inp, out):
                # Store output tensor
                self.activations[layer_name] = out
            return forward_hook

        for name, layer in self._target_layers:
            self._handles.append(layer.register_forward_hook(make_forward_hook(name)))

    def _remove_hooks(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def reinit(self, model, train_data=None):
        """
        Reinitialise after changing the model.
        """
        self._remove_hooks()
        self.model = model
        self._target_layers = self._auto_select_target_layers(self.model)
        self.activations = {}
        self._register_hooks()

    def _as_batch5(self, X: torch.Tensor) -> torch.Tensor:
        """Ensure shape [B, C, D, H, W]."""
        if X.dim() == 4:
            return X.unsqueeze(0)
        return X

    @torch.no_grad()
    def get_output(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(X)

    @torch.no_grad()
    def _infer_targets(self, X: torch.Tensor) -> torch.Tensor:
        """
        Choose target indices:
          - Binary / 1-logit: always index 0
          - Multiclass: argmax
        """
        self.model.eval()
        logits = self.model(X)

        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            B = logits.shape[0] if logits.ndim >= 1 else 1
            return torch.zeros(B, dtype=torch.long, device=X.device)

        return torch.argmax(logits, dim=1).long().to(X.device)

    def _targets_tensor(self, y, X: torch.Tensor) -> torch.Tensor:
        if y is None or self.use_predicted_labels:
            return self._infer_targets(X)

        y = torch.as_tensor(y, dtype=torch.long, device=X.device).view(-1)
        if y.numel() == 1 and X.shape[0] > 1:
            y = y.expand(X.shape[0])
        return y

    def _smooth_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Optional smoothing on [D,H,W] cam using avg pooling.
        """
        if not self.smooth_kernel:
            return cam
        k = self.smooth_kernel
        x = cam[None, None]  # [1,1,D,H,W]
        x = F.avg_pool3d(x, kernel_size=k, stride=1, padding=k // 2)
        return x[0, 0]

    def _compute_cam(
        self,
        activations: torch.Tensor,   # [1, C, d, h, w]
        gradients: torch.Tensor,     # [1, C, d, h, w]
        target_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Returns CAM as [D,H,W] (upsampled), with optional ReLU + smoothing.
        """
        if self.method == "gradcam":
            # weights: global average pooled gradients
            weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # [1,C,1,1,1]

        else:
            grad = gradients
            grad2 = grad.pow(2)
            grad3 = grad.pow(3)

            denom = 2.0 * grad2 + (activations * grad3).sum(dim=(2, 3, 4), keepdim=True)
            denom = denom + 1e-7

            alpha = grad2 / denom
            positive_grad = F.relu(grad)
            weights = (alpha * positive_grad).sum(dim=(2, 3, 4), keepdim=True)  # [1,C,1,1,1]

        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1,1,d,h,w]

        if self.apply_relu:
            cam = F.relu(cam)

        # Upsample to full patch size
        cam = F.interpolate(
            cam,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )  # [1,1,D,H,W]

        cam = cam[0, 0]  # [D,H,W]
        cam = self._smooth_cam(cam)
        return cam

    def gen_attr(self, X: torch.Tensor, y=None):
        """
        Compute 3D Grad-CAM/Grad-CAM++ attribution maps.

        Returns numpy array:
          - [D, H, W]       if X was a single volume
          - [B, D, H, W]    if X was a batch

        NOTE:
          Output is non-negative by default (apply_relu=True).
          Normalisation to [0,1] is done by eval pipeline (_prep_map).
        """
        Xb = self._as_batch5(X)
        device = Xb.device
        self.model.eval()

        B, C, D, H, W = Xb.shape
        cams = []

        with torch.enable_grad():
            Xb = Xb.to(device)
            y_t = self._targets_tensor(y, Xb)  # [B]

            for b in range(B):
                self.model.zero_grad(set_to_none=True)
                self.activations = {}

                x_single = Xb[b:b + 1].clone().detach().to(device)
                x_single.requires_grad_(True)

                logits = self.model(x_single)

                if len(self.activations) == 0:
                    raise RuntimeError(
                        "gradcam3d_atr: forward hook did not capture activations. "
                        "Check target layer selection."
                    )

                # Pick scalar score
                if logits.ndim == 1:
                    score = logits[0] if logits.numel() == 1 else logits[int(y_t[b].item())]
                elif logits.shape[1] == 1:
                    score = logits[0, 0]  # 1-logit binary classifier
                else:
                    score = logits[0, int(y_t[b].item())]

                # Capture gradients for each target layer activation tensor
                grads_container: dict[str, torch.Tensor] = {}
                act_hooks = []

                for name, _layer in self._target_layers:
                    if name not in self.activations:
                        continue
                    act = self.activations[name]

                    def _save_grad(grad, n=name):
                        grads_container[n] = grad

                    act_hooks.append(act.register_hook(_save_grad))

                score.backward(retain_graph=False)

                for h in act_hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass

                # Compute per-layer CAMs and fuse
                layer_cams = []
                for name, _layer in self._target_layers:
                    if name not in self.activations or name not in grads_container:
                        continue
                    act = self.activations[name].detach()
                    grad = grads_container[name].detach()

                    cam_l = self._compute_cam(act, grad, target_size=(D, H, W))

                    layer_cams.append(cam_l)


                if len(layer_cams) == 0:
                    raise RuntimeError("gradcam3d_atr: no CAMs computed (missing activations/gradients).")

                if len(layer_cams) == 1:
                    cam_b = layer_cams[0]
                else:
                    stack = torch.stack(layer_cams, dim=0)  # [L,D,H,W]
                    if self.fuse_layers == "mean":
                        cam_b = stack.mean(dim=0)
                    elif self.fuse_layers == "sum":
                        cam_b = stack.sum(dim=0)
                    else:  # max
                        cam_b = stack.max(dim=0).values

                cams.append(cam_b)

        cams_t = torch.stack(cams, dim=0)  # [B,D,H,W]
        attr = cams_t.detach().cpu().numpy().astype(np.float32)
        return attr.squeeze()

    def gen_mask(self, X: torch.Tensor):
        """
        Build a binary mask from CAM attributions (top-ratio per sample).
        Returns (mlogits, X*mask, mask).

        mask: [B, 1, D, H, W]
        """
        Xb = self._as_batch5(X)
        device = Xb.device

        # Use predicted labels for mask generation
        y_pred = self._infer_targets(Xb)
        attr = self.gen_attr(Xb, y_pred)  # np [B,D,H,W] or [D,H,W]

        if attr.ndim == 3:
            attr = attr[None, ...]

        B, D, H, W = attr.shape
        attr_t = torch.from_numpy(attr).to(device=device, dtype=torch.float32).view(B, 1, D, H, W)

        if self.normalise_for_mask:
            # per-sample min-max normalisation to make top-k threshold stable
            a = attr_t.view(B, -1)
            lo = a.min(dim=1, keepdim=True).values
            hi = a.max(dim=1, keepdim=True).values
            attr_t = ((a - lo) / (hi - lo + 1e-8)).view(B, 1, D, H, W)

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


class gradcam3d_ranked_atr(gradcam3d_atr):
    """
    Ranked version of 3D Grad-CAM/Grad-CAM++.

    Applies rank-based normalisation so that only voxel ordering matters.
    """

    def __init__(self, model, train_data=None, use_predicted_labels=False, **kwargs):
        super().__init__(model, train_data, use_predicted_labels, **kwargs)
        self.attr_name = "gradcam3d_ranked"

    def rank_attr(self, a: np.ndarray) -> np.ndarray:
        flat = a.flatten()
        order = np.argsort(flat)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, num=flat.size, endpoint=True)
        return ranks.reshape(a.shape)

    def gen_attr(self, X: torch.Tensor, y=None):
        a = super().gen_attr(X, y)
        if a.ndim == 3:
            return self.rank_attr(a)
        return np.stack([self.rank_attr(v) for v in a], axis=0)


__all__ = ["gradcam3d_atr", "gradcam3d_ranked_atr"]
