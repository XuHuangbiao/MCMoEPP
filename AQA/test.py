import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader

from .models.model import MoE_AQA

MASK_ROOT = Path(__file__).resolve().parent / "masks"
DEFAULT_RATES = tuple(round(index / 10, 1) for index in range(8))


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


@dataclass(frozen=True)
class Metrics:
    """AQA metrics in the dataset's original score range."""

    srcc: float
    mse: float


def load_availability_masks(sample_ids, dataset: str, action: str, missing_rate: float) -> torch.Tensor:
    """Read released [V, F, A] masks and return the model's [V, A, F] order."""
    if missing_rate not in DEFAULT_RATES:
        raise ValueError(f"unsupported missing rate: {missing_rate}")
    path = MASK_ROOT / f"{dataset}_{action}_rate{missing_rate:.1f}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    if set(stored) != set(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"mask/sample ID mismatch: {path}")
    masks = []
    for sample_id in sample_ids:
        mask = stored[sample_id]
        if len(mask) != 3 or any(value not in (0, 1) for value in mask) or not any(mask):
            raise ValueError(f"invalid mask for {sample_id!r}: {mask}")
        masks.append([mask[0], mask[2], mask[1]])
    return torch.tensor(masks, dtype=torch.long)


@torch.no_grad()
def evaluate_grouped_masks(
    network: MoE_AQA,
    loader: DataLoader,
    masks: torch.Tensor,
    device: torch.device,
    score_range: float,
) -> Metrics:
    """Evaluate per-sample masks by grouping identical masks per mini-batch."""

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    squared_error = 0.0
    total = 0
    offset = 0

    for video, audio, flow, label in loader:
        batch_size = label.shape[0]
        batch_masks = masks[offset : offset + batch_size]
        offset += batch_size
        video, audio, flow = video.to(device), audio.to(device), flow.to(device)
        label = label.float().to(device)
        batch_prediction = torch.empty(batch_size, dtype=label.dtype, device=device)
        for mask in torch.unique(batch_masks, dim=0):
            indices = torch.nonzero((batch_masks == mask).all(dim=1), as_tuple=False).squeeze(1)
            device_indices = indices.to(device)
            output = network(
                video.index_select(0, device_indices),
                audio.index_select(0, device_indices),
                flow.index_select(0, device_indices),
                mask.tolist(),
            )["output"]
            batch_prediction.index_copy_(0, device_indices, output)

        squared_error += torch.square((batch_prediction - label) * score_range).sum().item()
        total += batch_size
        predictions.append(batch_prediction.cpu().numpy())
        targets.append(label.cpu().numpy())

    if offset != len(masks):
        raise RuntimeError(f"mask/sample mismatch: consumed={offset}, masks={len(masks)}")
    srcc = spearmanr(np.concatenate(predictions), np.concatenate(targets)).statistic
    return Metrics(srcc=float(srcc), mse=squared_error / total)
