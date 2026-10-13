"""Validation and test evaluation for sentiment recognition."""

from __future__ import annotations

import numpy as np
import torch

from .metrics import compute_mae_corr_mse, compute_waf1_acc
from .missing_protocol import apply_modality_mask, actual_missing_rate, sample_utterance_mask


def _valid_values(tensor: torch.Tensor, utterance_mask: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().numpy()
    valid = utterance_mask.detach().cpu().numpy().astype(bool)
    return values[valid].reshape(-1)


def evaluate_epoch(epoch, model, data_loader, device, args):
    """Evaluate one split using the formal epoch-dependent stochastic masks."""
    model.eval()
    predictions = []
    labels = []
    realized_rates = []

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            text, visual, audio, label, utterance_mask = (item.to(device) for item in batch)
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + 2_000_000 + epoch * 100_000 + batch_index)
            modality_mask = sample_utterance_mask(
                text.size(0),
                text.size(1),
                args.missing_rate,
                device,
                generator=generator,
            )
            text, visual, audio = apply_modality_mask(text, visual, audio, modality_mask)
            prediction = model(text, visual, audio, modality_mask)["output"]
            predictions.append(_valid_values(prediction, utterance_mask))
            labels.append(_valid_values(label, utterance_mask))
            realized_rates.append(actual_missing_rate(modality_mask))

    prediction = np.concatenate(predictions) * 6.0 - 3.0
    label = np.concatenate(labels) * 6.0 - 3.0
    _has_zero, nonzero = compute_waf1_acc(prediction, label)
    regression = compute_mae_corr_mse(prediction, label)
    return {
        "waf1": float(nonzero["waf1"] or 0.0),
        "acc": float(nonzero["acc"] or 0.0),
        "mae": regression["mae"],
        "corr": regression["corr"],
        "mse": regression["mse"],
        "actual_missing": float(np.mean(realized_rates)),
    }
