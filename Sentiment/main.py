"""Formal CMU-MOSI/CMU-MOSEI training and evaluation entry point."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "Sentiment"

import csv
import json
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import options
from .datasets import CMUSentimentDataset
from .models import loss, model
from .test import evaluate_epoch
from .train import train_epoch

try:
    from tensorboardX import SummaryWriter
except ImportError:

    class SummaryWriter:  # pragma: no cover - used only without tensorboardX
        """No-op fallback so training does not require TensorBoard."""

        def __init__(self, *_args, **_kwargs):
            pass

        def add_scalar(self, *_args, **_kwargs):
            pass

        def close(self):
            pass


def setup_seed(seed: int) -> None:
    """Match the deterministic setup used by the formal experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@contextmanager
def preserve_training_rng(network: torch.nn.Module):
    """Keep validation from changing the next training epoch's random state."""
    module_modes = {module: module.training for module in network.modules()}
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for module, mode in module_modes.items():
            module.train(mode)


def validate_protocol(args) -> None:
    """Reject options that would silently diverge from the paper protocol."""
    if not 0.0 <= args.missing_rate <= 0.7:
        raise ValueError("--missing-rate must be between 0.0 and 0.7.")
    if args.warmup < 0 or args.warmup >= args.epoch:
        raise ValueError("--warmup must be non-negative and smaller than --epoch.")
    if args.test and not args.ckpt:
        raise ValueError("--test requires --ckpt.")


def resolve_device(value: Optional[str]) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def make_loader(split: str, args, shuffle: bool) -> DataLoader:
    dataset = CMUSentimentDataset(split=split, args=args)
    return DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    )


def build_model(args, device: torch.device) -> torch.nn.Module:
    network = model.MoE_Sentiment(
        args.in_dim,
        args.hidden_dim,
        args.n_head,
        args.n_encoder,
        args.n_decoder,
        args.n_query,
        args.dropout,
        args,
    )
    return network.to(device)


def build_optimizer(network: torch.nn.Module, args):
    common = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optim == "adam":
        return torch.optim.Adam(network.parameters(), **common)
    if args.optim == "adamw":
        return torch.optim.AdamW(network.parameters(), **common)
    if args.optim == "sgd":
        return torch.optim.SGD(network.parameters(), momentum=args.momentum, **common)
    return torch.optim.RMSprop(network.parameters(), momentum=args.momentum, **common)


def build_scheduler(optimizer, args):
    if args.lr_decay is None:
        return None
    if args.lr_decay == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epoch - args.warmup,
            eta_min=args.lr * args.decay_rate,
        )
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(args.epoch - 30, 1)],
        gamma=args.decay_rate,
    )


def checkpoint_path(args) -> Path:
    return Path(args.output_dir) / "checkpoints" / f"{args.model_name}_best.pt"


def resolve_checkpoint(value: str, args) -> Path:
    supplied = Path(value).expanduser()
    candidates = [supplied, Path(args.output_dir) / "checkpoints" / supplied]
    if not supplied.suffix:
        candidates.extend(
            Path(args.output_dir) / "checkpoints" / f"{supplied}{suffix}"
            for suffix in (".pt", ".pth", ".pkl")
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {value}")


def save_checkpoint(path: Path, network: torch.nn.Module, args, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": network.state_dict(),
            "epoch": epoch,
            "validation_metrics": metrics,
            "selection_metric": args.metric_main,
            "dataset": args.dataset,
            "missing_rate": args.missing_rate,
        },
        path,
    )


def load_checkpoint(path: Path, network: torch.nn.Module, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    network.load_state_dict(state_dict, strict=True)
    return payload if isinstance(payload, dict) else {}


def write_summary(
    args, best_epoch: int, validation: dict, test: dict, used_checkpoint: Path
) -> Path:
    output = Path(args.output_dir) / "summaries" / f"{args.model_name}_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "selection_metric": args.metric_main,
        "best_epoch": best_epoch,
        "val_waf1": validation.get("waf1", ""),
        "val_acc": validation.get("acc", ""),
        "test_waf1": test["waf1"],
        "test_acc": test["acc"],
        "test_mae": test["mae"],
        "test_corr": test["corr"],
        "test_mse": test["mse"],
        "actual_missing": test["actual_missing"],
        "checkpoint": str(used_checkpoint.resolve()),
    }
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"Summary: {output.resolve()}")
    return output


def print_metrics(prefix: str, metrics: dict) -> None:
    print(
        f"{prefix} WAF1/Acc={metrics['waf1'] * 100:.2f}/{metrics['acc'] * 100:.2f} "
        f"actual_missing={metrics['actual_missing']:.4f}",
        flush=True,
    )


def run_test(args, device: torch.device) -> None:
    test_loader = make_loader("test", args, shuffle=False)
    network = build_model(args, device)
    path = resolve_checkpoint(args.ckpt, args)
    metadata = load_checkpoint(path, network, device)
    if args.test_eval_epoch is not None:
        best_epoch = args.test_eval_epoch
    elif "epoch" in metadata:
        best_epoch = int(metadata["epoch"])
    else:
        raise ValueError(
            "Checkpoint is missing the selected validation epoch; "
            "supply --test-eval-epoch for a bare state dict."
        )
    if "selection_metric" in metadata:
        args.metric_main = metadata["selection_metric"]
    metrics = evaluate_epoch(best_epoch, network, test_loader, device, args)
    print_metrics("Test", metrics)
    write_summary(args, best_epoch, metadata.get("validation_metrics", {}), metrics, path)


def run_training(args, device: torch.device) -> None:
    train_loader = make_loader("train", args, shuffle=True)
    test_loader = make_loader("test", args, shuffle=False)
    valid_loader = make_loader("valid", args, shuffle=False)
    next(iter(test_loader))  # Consume one batch to preserve the RNG state used by the reported runs.
    network = build_model(args, device)
    criterion = loss.LossFun(args.alpha, args.margin)
    optimizer = build_optimizer(network, args)
    scheduler = build_scheduler(optimizer, args)
    warmup = None
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: step / args.warmup,
        )

    log_dir = Path(args.output_dir) / "logs" / args.model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "arguments.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    logger = SummaryWriter(str(log_dir))

    best_score = -float("inf")
    best_epoch = -1
    best_validation = {}
    stale_epochs = 0
    path = checkpoint_path(args)

    for epoch in range(args.epoch):
        if warmup is not None and epoch < args.warmup:
            warmup.step()
        train_loss, train_corr, train_missing = train_epoch(
            epoch,
            network,
            criterion,
            train_loader,
            optimizer,
            logger,
            device,
            args,
        )
        if scheduler is not None and (args.lr_decay != "cos" or epoch >= args.warmup):
            scheduler.step()

        with preserve_training_rng(network):
            validation = evaluate_epoch(epoch, network, valid_loader, device, args)
        logger.add_scalar("valid/waf1", validation["waf1"], epoch)
        logger.add_scalar("valid/acc", validation["acc"], epoch)
        current_score = validation[args.metric_main]
        improved = current_score > best_score
        if improved:
            best_score = current_score
            best_epoch = epoch
            best_validation = dict(validation)
            stale_epochs = 0
            save_checkpoint(path, network, args, epoch, validation)
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d} train_loss={float(train_loss):.6f} "
            f"train_corr={train_corr:.4f} train_missing={train_missing:.4f}",
            flush=True,
        )
        print_metrics("Validation", validation)
        if args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(
                f"Early stop at epoch {epoch}; best validation epoch is {best_epoch}.",
                flush=True,
            )
            break

    logger.close()
    load_checkpoint(path, network, device)
    test_metrics = evaluate_epoch(best_epoch, network, test_loader, device, args)
    print_metrics(f"Best validation (selected by {args.metric_main.upper()})", best_validation)
    print_metrics("Final test", test_metrics)
    write_summary(args, best_epoch, best_validation, test_metrics, path)


def main() -> None:
    args = options.parser.parse_args()
    validate_protocol(args)
    setup_seed(args.seed)
    device = resolve_device(args.device)
    if args.test:
        run_test(args, device)
    else:
        run_training(args, device)


if __name__ == "__main__":
    main()
