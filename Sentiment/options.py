"""Command-line options for MOSI/MOSEI experiments."""

from __future__ import annotations

import argparse
from typing import Union


def parse_bool(value: Union[str, bool]) -> bool:
    """Parse explicit boolean values without treating ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or test MCMoE++ on CMU-MOSI/CMU-MOSEI with utterance-level missing modalities."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", choices=("MOSI", "MOSEI"), required=True)
    data.add_argument("--text-path", required=True, help="Text feature directory.")
    data.add_argument("--video-path", required=True, help="Video feature directory.")
    data.add_argument("--audio-path", required=True, help="Audio feature directory.")
    data.add_argument(
        "--label-path", required=True,
        help="Label file containing the train, valid, and test splits.",
    )
    data.add_argument("--batch", type=int, default=32)
    data.add_argument("--num-workers", type=int, default=0)

    protocol = parser.add_argument_group("missing protocol")
    protocol.add_argument(
        "--missing-rate",
        type=float,
        choices=tuple(index / 10 for index in range(8)),
        required=True,
    )
    protocol.add_argument("--metric-main", choices=("waf1", "acc"), default="waf1")
    model = parser.add_argument_group("model")
    model.add_argument("--in_dim", type=int, default=768)
    model.add_argument("--hidden_dim", type=int, default=256)
    model.add_argument("--n_head", type=int, default=2)
    model.add_argument("--n_encoder", type=int, default=3)
    model.add_argument("--n_decoder", type=int, default=3)
    model.add_argument("--n_query", type=int, default=4)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--use_pe", type=parse_bool, default=False)

    training = parser.add_argument_group("optimization")
    training.add_argument("--epoch", type=int, default=400)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--optim", choices=("adam", "adamw", "sgd", "rmsprop"), default="adam")
    training.add_argument("--momentum", type=float, default=0.9)
    training.add_argument("--lr-decay", choices=("cos", "multistep"), default=None)
    training.add_argument("--decay-rate", type=float, default=0.1)
    training.add_argument("--warmup", type=int, default=0)
    training.add_argument("--early-stop-patience", type=int, default=0)
    training.add_argument("--alpha", type=float, default=1.0)
    training.add_argument("--margin", type=float, default=1.0)
    training.add_argument("--mse-loss-weight", type=float, default=10.0)
    training.add_argument("--recon-loss-weight", type=float, default=1.0)
    training.add_argument("--sign-loss-weight", type=float, default=0.0)

    output = parser.add_argument_group("output")
    output.add_argument("--model-name", required=True)
    output.add_argument("--output-dir", default="outputs")
    output.add_argument("--ckpt", default=None)
    output.add_argument(
        "--test-eval-epoch", type=int, default=None,
        help="Validation-selected epoch for a bare state-dict checkpoint.",
    )
    output.add_argument("--test", action="store_true")
    output.add_argument("--device", default=None, help="For example cuda, cuda:1, or cpu.")
    return parser


parser = build_parser()
