"""Evaluate AQA checkpoints under fixed or proportional modality missingness.

The model uses ``[V, A, F]``. Fixed evaluation covers the seven valid masks.
Rate evaluation reads the released per-video JSON masks in ``[V, F, A]`` order.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "AQA"

import argparse
import csv
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .datasets import FisVDataset, FS1000Dataset, RGDataset
from .models.model import MoE_AQA

FIXED_MASKS: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("VAF", (1, 1, 1)),
    ("VA", (1, 1, 0)),
    ("VF", (1, 0, 1)),
    ("AF", (0, 1, 1)),
    ("V", (1, 0, 0)),
    ("A", (0, 1, 0)),
    ("F", (0, 0, 1)),
)
DEFAULT_RATES = tuple(round(index / 10, 1) for index in range(8))
MASK_ROOT = Path(__file__).resolve().parent / "masks"


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


def _dataset(args: argparse.Namespace):
    classes = {"FS1000": FS1000Dataset, "FisV": FisVDataset, "RG": RGDataset}
    config = SimpleNamespace(score_range=None)
    dataset = classes[args.dataset](
        args.video_path,
        args.audio_path,
        args.flow_path,
        args.label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
        train=False,
        args=config,
    )
    return dataset, float(config.score_range)


def _model(args: argparse.Namespace, device: torch.device) -> MoE_AQA:
    model_args = SimpleNamespace(
        use_pe=args.use_pe,
        ablation_variant=args.ablation_variant,
        recon_layers=args.recon_layers,
    )
    network = MoE_AQA(
        args.in_dim,
        args.hidden_dim,
        args.n_head,
        args.n_encoder,
        args.n_decoder,
        args.n_query,
        args.dropout,
        model_args,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    network.load_state_dict(checkpoint, strict=True)
    network.eval()
    return network


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


def _write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("fixed", "rate"), required=True)
    parser.add_argument("--dataset", choices=("FS1000", "FisV", "RG"), required=True)
    parser.add_argument("--action-type", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--flow-path", required=True)
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-num", type=int, required=True)
    parser.add_argument("--in-dim", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-encoder", type=int, default=3)
    parser.add_argument("--n-decoder", type=int, default=3)
    parser.add_argument("--n-query", type=int, default=4)
    parser.add_argument("--recon-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-pe", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument(
        "--ablation-variant",
        choices=MoE_AQA.ABLATION_VARIANTS,
        default="full",
        help="Must match the architecture used to train the checkpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, score_range = _dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    network = _model(args, device)
    rows: List[Dict[str, object]] = []

    if args.protocol == "fixed":
        for name, mask in FIXED_MASKS:
            masks = torch.tensor(mask, dtype=torch.long).repeat(len(dataset), 1)
            metrics = evaluate_grouped_masks(network, loader, masks, device, score_range)
            rows.append(
                {
                    "dataset": args.dataset,
                    "action": args.action_type,
                    "condition": name,
                    "mask": "".join(str(value) for value in mask),
                    "srcc": metrics.srcc,
                    "mse": metrics.mse,
                }
            )
    else:
        for rate in args.rates:
            sample_ids = [sample_id for sample_id, _ in dataset.labels]
            masks = load_availability_masks(sample_ids, args.dataset, args.action_type, rate)
            metrics = evaluate_grouped_masks(network, loader, masks, device, score_range)
            rows.append(
                {
                    "dataset": args.dataset,
                    "action": args.action_type,
                    "requested_rate": rate,
                    "actual_rate": 1.0 - masks.float().mean().item(),
                    "mask_file": f"{args.dataset}_{args.action_type}_rate{rate:.1f}.json",
                    "srcc": metrics.srcc,
                    "mse": metrics.mse,
                }
            )

    _write_rows(args.output, rows)
    for row in rows:
        print(row)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
