import argparse


def parse_bool(value):
    """Parse explicit boolean CLI values without Python's ``bool('false')`` trap."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


parser = argparse.ArgumentParser()

parser.add_argument("--video-path", type=str, default="./dataset/FS1000/output_feature_fs1000_new")
parser.add_argument("--audio-path", type=str, default="./dataset/FS1000/ast_feature_fs1000_new")
parser.add_argument("--flow-path", type=str, default="./dataset/FS1000/i3d_avg_clip8_5s_fs1000")
parser.add_argument("--clip-num", type=int, default=95)

parser.add_argument("--dataset", type=str, choices=["FS1000", "FisV", "RG"], default="FS1000")
parser.add_argument("--train-label-path", type=str, default="./dataset/FS1000/train_fs1000_new.txt")
parser.add_argument("--test-label-path", type=str, default="./dataset/FS1000/val_fs1000_new.txt")

parser.add_argument("--action-type", type=str, default="TES")
parser.add_argument("--score-type", type=str, default="Total_Score")

parser.add_argument(
    "--model-name", type=str, default="action_net", help="name used to save model and logs"
)
parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument(
    "--resume",
    default=None,
    help="training-state checkpoint to resume (restores model, optimizer, schedulers, RNG, and epoch)",
)
parser.add_argument("--test", action="store_true", help="only evaluate, don't train")
parser.add_argument(
    "--missing-ratio",
    type=float,
    choices=tuple(index / 10 for index in range(8)),
    default=None,
    help="evaluate one proportional missing rate using the released masks/*.json",
)
parser.add_argument(
    "--cf_variant",
    choices=(
        "full",
        "baseline",
        "stem",
        "agmg",
        "specific_only_analysis",
        "no_ma_router",
        "no_anchor_residual",
        "no_shared_expert",
        "no_acdg",
    ),
    default="full",
    help="same-checkpoint inference intervention; non-full variants are rejected during training",
)
parser.add_argument(
    "--ablation_variant",
    choices=(
        "full",
        "baseline",
        "stem",
        "agmg",
        "ma_moe",
        "no_stem",
        "no_agmg",
        "no_ma_moe",
        "no_ma_router",
        "no_anchor_residual",
        "no_shared_expert",
        "no_acdg",
        "no_lrecon",
        "no_lalign",
        "no_ldiv",
    ),
    default="full",
    help="retrained ablation architecture; valid for both training and testing",
)

parser.add_argument("--epoch", type=int, default=400)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--optim", type=str, default="adam")

parser.add_argument("--lr-decay", type=str, default=None, help="use what decay scheduler")
parser.add_argument("--decay-rate", type=float, default=0.1, help="lr decay rate")
parser.add_argument("--warmup", type=int, default=0, help="warmup epoch")

parser.add_argument("--in_dim", type=int, default=768)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--n_head", type=int, default=2)
parser.add_argument("--n_encoder", type=int, default=3)
parser.add_argument("--n_decoder", type=int, default=3)
parser.add_argument("--n_query", type=int, default=4)
parser.add_argument("--recon-layers", type=int, default=2, help="number of independent completion cross-attention layers")
parser.add_argument("--use_pe", type=parse_bool, default=False)

parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--margin", type=float, default=1.0)
parser.add_argument("--mse-loss-weight", type=float, default=10.0)
parser.add_argument("--recon-loss-weight", type=float, default=1.0)

parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--score_range", type=int, default=None)
parser.add_argument("--target-missing-coef", type=float, default=None)
parser.add_argument("--target-missing-mse", type=float, default=None)
parser.add_argument("--target-save-prefix", type=str, default="target_pass")
parser.add_argument(
    "--checkpoint-selection",
    choices=("missing_avg", "vaf_coef", "vaf_mse", "fixed_epoch"),
    default="missing_avg",
    help="checkpoint rule; formal retrained ablations use the action-specific VAF/fixed-epoch rule",
)
parser.add_argument(
    "--selection-epoch",
    type=int,
    default=None,
    help="zero-based epoch saved when --checkpoint-selection=fixed_epoch",
)
parser.add_argument(
    "--resume-save-interval",
    type=int,
    default=1,
    help="save resumable training state every N epochs (model selection is unaffected)",
)
