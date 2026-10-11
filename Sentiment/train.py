"""One training epoch for the sentiment protocol."""

from __future__ import annotations

import numpy as np
import torch

from .missing_protocol import actual_missing_rate, sample_utterance_mask
from .utils import AverageMeter


def _valid_values(tensor: torch.Tensor, utterance_mask: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()[utterance_mask.detach().cpu().numpy().astype(bool)]


def train_epoch(epoch, model, loss_fn, train_loader, optimizer, logger, device, args):
    """Train with the same requested missing rate used by evaluation."""
    model.train()
    losses = AverageMeter("loss", logger)
    predictions = []
    labels = []
    realized_rates = []

    for batch_index, batch in enumerate(train_loader):
        text, visual, audio, label, utterance_mask = (item.to(device) for item in batch)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + epoch * 100_000 + batch_index)
        modality_mask = sample_utterance_mask(
            text.size(0),
            text.size(1),
            args.missing_rate,
            device,
            generator=generator,
        )
        output = model(text, visual, audio, modality_mask)
        prediction = output["output"]
        loss = loss_fn(
            prediction,
            label,
            output["embed"],
            output["embed2"],
            output["recon_loss"],
            args,
            sample_mask=utterance_mask,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sample_count = int(utterance_mask.sum().item())
        losses.update(loss, sample_count)
        predictions.append(_valid_values(prediction, utterance_mask))
        labels.append(_valid_values(label, utterance_mask))
        realized_rates.append(actual_missing_rate(modality_mask))

    predictions_np = np.concatenate(predictions)
    labels_np = np.concatenate(labels)
    correlation = 0.0
    if predictions_np.size > 1 and predictions_np.std() > 0 and labels_np.std() > 0:
        correlation = float(np.corrcoef(predictions_np, labels_np)[0, 1])
    average_loss = losses.done(epoch)
    if logger is not None:
        logger.add_scalar("train/correlation", correlation, epoch)
        logger.add_scalar("train/actual_missing", float(np.mean(realized_rates)), epoch)
    return average_loss, correlation, float(np.mean(realized_rates))
