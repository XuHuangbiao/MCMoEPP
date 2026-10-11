import torch
import torch.nn.functional as F
from torch import nn

from .triplet_loss import HardTripletLoss


class LossFun(nn.Module):
    def __init__(self, alpha, margin):
        super(LossFun, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha
        self.alpha_recon = 1.0

    def forward(self, pred, label, feat, feat2, recon_loss, args, sample_mask=None):
        # feat (b, n, c), x (b, t, c)
        if feat is not None:
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.reshape(-1, c)  # (bn, c)
            la = torch.arange(n, device=device).repeat(b)

            t_loss = self.triplet_loss(flat_feat, la)
            if feat2 is not None:
                device = feat2.device
                b, n, c = feat2.shape
                flat_feat2 = feat2.reshape(-1, c)  # (bn, c)
                la2 = torch.arange(n, device=device).repeat(b)

                t_loss += self.triplet_loss(flat_feat2, la2)
        else:
            self.alpha = 0
            t_loss = 0
        if sample_mask is None:
            mse_weight = float(getattr(args, "mse_loss_weight", 10.0))
            mse_loss = mse_weight * self.mse_loss(pred, label)
            valid_mask = torch.ones_like(label, dtype=torch.bool)
        else:
            mse_weight = float(getattr(args, "mse_loss_weight", 10.0))
            mask = sample_mask.to(device=pred.device, dtype=pred.dtype)
            while mask.dim() < pred.dim():
                mask = mask.unsqueeze(-1)
            diff = (pred - label) ** 2
            mse_loss = mse_weight * (diff * mask).sum() / mask.sum().clamp_min(1.0)
            valid_mask = mask > 0.5

        sign_weight = float(getattr(args, "sign_loss_weight", 0.0))
        sign_loss = pred.new_tensor(0.0)
        if sign_weight > 0 and getattr(args, "dataset", None) in ("MOSI", "MOSEI"):
            pred_flat = pred.reshape(-1)
            label_flat = label.reshape(-1)
            valid_flat = valid_mask.reshape(-1) & (torch.abs(label_flat - 0.5) > 1e-6)
            if valid_flat.any():
                sign_pred = pred_flat[valid_flat].clamp(1e-6, 1.0 - 1e-6)
                sign_target = (label_flat[valid_flat] > 0.5).to(dtype=pred.dtype)
                sign_loss = F.binary_cross_entropy(sign_pred, sign_target)

        recon_weight = float(getattr(args, "recon_loss_weight", self.alpha_recon))
        return mse_loss + self.alpha * t_loss + recon_weight * recon_loss + sign_weight * sign_loss
