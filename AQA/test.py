import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn


def test_epoch(epoch, model, test_loader, logger, device, mask, args):
    mse_loss = nn.MSELoss().to(device)
    model.eval()
    preds = np.array([])
    labels = np.array([])
    tol_loss, tol_sample = 0, 0

    with torch.no_grad():
        for video_feat, audio_feat, flow_feat, label in test_loader:
            video_feat = video_feat.to(device)
            audio_feat = audio_feat.to(device)  # (b, t, c)
            flow_feat = flow_feat.to(device)  # (b, t, c)
            label = label.float().to(device)
            out = model(video_feat, audio_feat, flow_feat, mask)
            pred = out["output"]

            loss = mse_loss(pred * args.score_range, label * args.score_range)
            tol_loss += loss.item() * label.shape[0]
            tol_sample += label.shape[0]

            if len(preds) == 0:
                preds = pred.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
                labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)
    avg_coef, _ = spearmanr(preds, labels)
    avg_loss = float(tol_loss) / float(tol_sample)

    if logger is not None:
        logger.add_scalar("Test coef", avg_coef, epoch)
        logger.add_scalar("Test loss", avg_loss, epoch)
    return avg_loss, avg_coef
