import torch
from torch import nn

from .triplet_loss import HardTripletLoss


class LossFun(nn.Module):
    def __init__(self, alpha, margin, mse_weight=10.0, recon_weight=1.0):
        super(LossFun, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.alpha = alpha
        self.mse_weight = float(mse_weight)
        self.alpha_recon = float(recon_weight)

    def forward(self, pred, label, feat, feat2, recon_loss, args):
        # feat (b, n, c), x (b, t, c)
        use_diversity = getattr(args, "ablation_variant", "full") != "no_ldiv"
        if feat is not None and use_diversity:
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
            t_loss = pred.new_zeros(())
        mse_loss = self.mse_weight * self.mse_loss(pred, label)
        return mse_loss + self.alpha * t_loss + self.alpha_recon * recon_loss
