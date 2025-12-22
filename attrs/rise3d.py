import numpy as np
import torch
import torch.nn.functional as F

from .attribution_template import attribution_template_class


class rs3d_atr(attribution_template_class):
    """
    3D RISE (Randomized Input Sampling for Explanation).

    - Black-box: only uses model outputs (no gradients).
    - Works for binary (1-logit) and multiclass (K-logit) models.
    - Generates random 3D masks at a low resolution and upsamples them
      to the full patch size (D, H, W) with trilinear interpolation.
    - Aggregates masks weighted by the model's score for the target class.

    """

    def __init__(self, model, train_data=None, use_predicted_labels=False):
        self.attr_name = "rise3d"
        self.use_predicted_labels = use_predicted_labels
        self.model = model

        # RISE hyperparameters
        self.n_masks = 8192
        self.initial_mask_shape = (4, 6, 6)
        self.mask_prob = 0.5 
        self.internal_batch_size = 32

        # For deletion/insertion-style experiments
        self.mask_top_ratio = 0.15

    def reinit(self, model, train_data=None):
        self.model = model

    def _as_batch5(self, X: torch.Tensor) -> torch.Tensor:
        """
        Ensure input is [B, C, D, H, W].
        Accepts [C, D, H, W] and adds a batch dimension.
        """
        if X.dim() == 4:
            return X.unsqueeze(0)
        return X

    @torch.no_grad()
    def get_output(self, X: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for model forward.
        """
        self.model.eval()
        return self.model(X)

    @torch.no_grad()
    def _infer_targets(self, X: torch.Tensor) -> torch.Tensor:
        """
        Choose target indices for RISE:

          - Binary / 1-logit model: always attribute w.r.t. that single logit (index 0).
          - Multiclass: attribute w.r.t. argmax class.
        """
        self.model.eval()
        logits = self.model(X)

        # Binary / single-logit case
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            B = logits.shape[0] if logits.ndim >= 1 else 1
            return torch.zeros(B, dtype=torch.long, device=X.device)

        # Multiclass case
        return torch.argmax(logits, dim=1).long().to(X.device)

    def _targets_tensor(self, y, X: torch.Tensor) -> torch.Tensor:
        """
        Resolve targets based on y and self.use_predicted_labels.
        """
        if y is None or self.use_predicted_labels:
            return self._infer_targets(X)

        y = torch.as_tensor(y, dtype=torch.long, device=X.device).view(-1)
        if y.numel() == 1 and X.shape[0] > 1:
            y = y.expand(X.shape[0])
        return y

    def _generate_masks(self, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Generate 3D random masks:

          1. Sample low-res Bernoulli masks with shape (d0, h0, w0).
          2. Upsample with trilinear interpolation to (D, H, W).

        Returns:
          masks: [N, 1, D, H, W] float32 in [0, 1]
        """
        d0, h0, w0 = self.initial_mask_shape
        N = self.n_masks

        # Step 1: low-res binary masks
        masks_small = (torch.rand(N, 1, d0, h0, w0, device=device) < self.mask_prob).float()

        # Step 2: upsample to full volume size (trilinear for 3D)
        masks = F.interpolate(
            masks_small,
            size=(D, H, W),
            mode="trilinear",
            align_corners=False,
        )

        return masks  # [N, 1, D, H, W] in [0,1]

    @torch.no_grad()
    def gen_attr(self, X: torch.Tensor, y=None):
        """
        Compute 3D RISE attribution maps.

        Returns numpy array:
          - [D, H, W]      if X was a single volume
          - [B, D, H, W]   if X was a batch
        """
        Xb = self._as_batch5(X)
        device = Xb.device
        self.model.eval()

        B, C, D, H, W = Xb.shape
        y_t = self._targets_tensor(y, Xb)  # [B]
        masks = self._generate_masks(D, H, W, device=device)  # [N, 1, D, H, W]
        N = masks.shape[0]
        P = D * H * W

        # Output container
        attr_maps = torch.zeros((B, D, H, W), device=device, dtype=torch.float32)

        for b in range(B):
            target_b = int(y_t[b].item())

            # Repeat the b-th sample for all masks: [N, C, D, H, W]
            x_b = Xb[b:b + 1].expand(N, -1, -1, -1, -1)

            # Flat accumulator in (D*H*W)
            attr_flat = torch.zeros(P, device=device, dtype=torch.float32)

            # Process masks in chunks to avoid OOM
            for start in range(0, N, self.internal_batch_size):
                end = min(N, start + self.internal_batch_size)
                m_chunk = masks[start:end]                  # [M, 1, D, H, W]
                x_chunk = x_b[start:end] * m_chunk          # [M, C, D, H, W]

                logits_chunk = self.model(x_chunk)  # [M], [M,1], or [M,K]

                if logits_chunk.ndim == 1:
                    # [M] single logit -> sigmoid
                    scores = torch.sigmoid(logits_chunk)              # [M]
                elif logits_chunk.shape[1] == 1:
                    # [M,1] single logit -> sigmoid
                    scores = torch.sigmoid(logits_chunk[:, 0])        # [M]
                else:
                    # [M,K] multiclass -> softmax prob for target_b
                    probs  = torch.softmax(logits_chunk, dim=1)       # [M,K]
                    targets = torch.full(
                        (logits_chunk.size(0),),
                        target_b,
                        dtype=torch.long,
                        device=device,
                    )
                    scores = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # [M]

                # Flatten masks to [M, P]
                m_flat = m_chunk[:, 0].reshape(scores.size(0), -1)

                attr_flat += scores @ m_flat  # [P]

            # Normalise by number of masks and expected mask value (mask_prob)
            attr_flat /= (N * self.mask_prob + 1e-8)
            attr_maps[b] = attr_flat.view(D, H, W)

        attr = attr_maps.detach().cpu().numpy().astype(np.float32)
        return attr.squeeze()

    @torch.no_grad()
    def gen_mask(self, X: torch.Tensor):
        """
        Build a binary mask from RISE attributions (top-ratio per sample).
        Returns (mlogits, X*mask, mask).
        """
        Xb = self._as_batch5(X)
        device = Xb.device

        # Use predicted labels for mask generation (like IG)
        y_pred = self._infer_targets(Xb)
        attr = self.gen_attr(Xb, y_pred)  # np [B, D, H, W] or [D, H, W]

        if attr.ndim == 3:
            attr = attr[None, ...]  # [1, D, H, W]

        B, D, H, W = attr.shape
        attr_t = torch.from_numpy(attr).to(device=device, dtype=torch.float32).view(
            B, 1, D, H, W
        )

        masks = []
        for b in range(B):
            flat = attr_t[b].flatten()
            k = max(1, int(self.mask_top_ratio * flat.numel()))
            if k >= flat.numel():
                thr = flat.min()
            else:
                thr = torch.topk(flat, k).values.min()
            masks.append((attr_t[b] >= thr).float())

        mask = torch.stack(masks, dim=0)  # [B, 1, D, H, W]
        x_masked = Xb * mask
        mlogits = self.get_output(x_masked)
        return mlogits, x_masked, mask


class rs3d_ranked_atr(rs3d_atr):
    """
    Ranked version of 3D RISE.

    Applies rank-based normalization to the attribution maps so that
    only the ordering of voxels matters (useful for some evaluation metrics).
    """

    def __init__(self, model, train_data=None, use_predicted_labels=False):
        super().__init__(model, train_data, use_predicted_labels)
        self.attr_name = "rise3d_ranked"

    def gen_attr(self, X: torch.Tensor, y=None):
        a = super().gen_attr(X, y)
        if a.ndim == 3:
            return self.rank_attr(a)
        return np.stack([self.rank_attr(v) for v in a], axis=0)


__all__ = ["rs3d_atr", "rs3d_ranked_atr"]
