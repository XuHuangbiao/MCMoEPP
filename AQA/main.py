import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "AQA"

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import options, train
from .datasets import FisVDataset, FS1000Dataset, RGDataset

try:
    from tensorboardX import SummaryWriter
except ImportError:  # TensorBoard logging is optional for evaluation/help paths.
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:

        class SummaryWriter:  # pragma: no cover - minimal no-op fallback
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass


import math
import os
import random
from contextlib import contextmanager
from datetime import datetime

from .models import loss, model
from .test import test_epoch


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def get_optim(model, args):
    if args.optim == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    if args.optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optim == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    raise Exception("Unknown optimizer")


def get_scheduler(optim, args):
    if args.lr_decay is None:
        return None

    if args.lr_decay == "cos":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate
        )

    if args.lr_decay == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=[args.epoch - 30], gamma=args.decay_rate
        )

    raise Exception("Unknown Scheduler")


def compute_average(metric_list):
    """
    metric_list: [(coef, mse), ...]
    coef uses Fisher-Z average; mse uses arithmetic average.
    """
    r_values = [x[0] for x in metric_list]
    z_list = []

    for r in r_values:
        r = float(np.clip(r, -0.999999, 0.999999))
        z_list.append(0.5 * (math.log(1 + r) - math.log(1 - r)))

    zz = np.mean(z_list)
    coef_avg = (np.exp(zz) - np.exp(-zz)) / (np.exp(zz) + np.exp(-zz))
    mse_avg = np.mean([x[1] for x in metric_list])

    return coef_avg, mse_avg


@contextmanager
def preserve_rng_and_model_mode(model):
    """
    Ensure extra evaluation does not change subsequent training randomness.
    """
    module_modes = {module: module.training for module in model.modules()}

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [s.clone() for s in torch.cuda.get_rng_state_all()]

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

        for module, mode in module_modes.items():
            module.train(mode)


def test_incomplete_modalities(epoch, model, test_loader, device, args):
    incomplete_combinations = [
        [1, 1, 0],  # VA
        [1, 0, 1],  # VF
        [0, 1, 1],  # AF
        [1, 0, 0],  # V
        [0, 0, 1],  # F
        [0, 1, 0],  # A
    ]

    modalities = ["V", "A", "F"]
    results = []
    results_for_avg = []

    for mask in incomplete_combinations:
        name = "".join([modalities[j] for j in range(3) if mask[j] == 1])
        test_loss, coef = test_epoch(epoch, model, test_loader, None, device, mask, args)
        results.append(
            {
                "name": name,
                "mask": mask,
                "coef": coef,
                "mse": test_loss,
            }
        )
        results_for_avg.append((coef, test_loss))

    avg_coef, avg_loss = compute_average(results_for_avg)
    return avg_loss, avg_coef, results


def append_missing_result(
    log_path, epoch, args, full_coef, full_loss, missing_avg_loss, missing_avg_coef, missing_results
):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Dataset: {args.dataset}\tAction: {args.action_type}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Full VAF: coef={full_coef:.4f}, mse={full_loss:.4f}\n")
        f.write("-" * 90 + "\n")
        for item in missing_results:
            f.write(
                f"{item['name']:>3s} mask={item['mask']} coef={item['coef']:.4f} mse={item['mse']:.4f}\n"
            )
        f.write("-" * 90 + "\n")
        f.write(f"Incomplete Average: coef={missing_avg_coef:.4f}, mse={missing_avg_loss:.4f}\n")
        f.write("=" * 90 + "\n\n")


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_checkpoint_path(value):
    if os.path.isfile(value):
        return value
    candidate = os.path.join("./ckpt", value)
    if os.path.isfile(candidate):
        return candidate
    if not value.endswith(".pt") and not value.endswith(".pkl"):
        for extension in (".pt", ".pkl"):
            candidate = os.path.join("./ckpt", value + extension)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(f"checkpoint not found: {value}")


if __name__ == "__main__":
    args = options.parser.parse_args()
    if args.ablation_variant != "full" and args.cf_variant != "full":
        raise ValueError("--ablation_variant and --cf_variant cannot both be non-full")
    if args.checkpoint_selection == "fixed_epoch":
        if args.selection_epoch is None:
            raise ValueError("--selection-epoch is required for fixed_epoch selection")
        if not 0 <= args.selection_epoch < args.epoch:
            raise ValueError("--selection-epoch must satisfy 0 <= selection_epoch < epoch")
    elif args.selection_epoch is not None:
        raise ValueError("--selection-epoch is only valid with fixed_epoch selection")
    if args.resume_save_interval < 1:
        raise ValueError("--resume-save-interval must be at least 1")
    setup_seed(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    Dataset = FS1000Dataset
    if args.dataset == "RG":
        Dataset = RGDataset
    elif args.dataset == "FisV":
        Dataset = FisVDataset
    elif args.dataset == "FS1000":
        Dataset = FS1000Dataset

    train_data = Dataset(
        args.video_path,
        args.audio_path,
        args.flow_path,
        args.train_label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
        args=args,
    )

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=8)
    print(len(train_data))

    test_data = Dataset(
        args.video_path,
        args.audio_path,
        args.flow_path,
        args.test_label_path,
        clip_num=args.clip_num,
        action_type=args.action_type,
        train=False,
        args=args,
    )
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=8)
    print("=============Load dataset successfully=============")

    model = model.MoE_AQA(
        args.in_dim,
        args.hidden_dim,
        args.n_head,
        args.n_encoder,
        args.n_decoder,
        args.n_query,
        args.dropout,
        args,
    ).to(device)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")
    print(f"Total learnable items: {len(enabled)}")

    loss_fn = loss.LossFun(
        args.alpha,
        args.margin,
        mse_weight=args.mse_loss_weight,
        recon_weight=args.recon_loss_weight,
    )
    train_fn = train.train_epoch

    if args.ckpt is not None and args.resume is not None:
        raise ValueError("--ckpt and --resume cannot be used together")
    if args.ckpt is not None:
        checkpoint_path = resolve_checkpoint_path(args.ckpt)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            checkpoint = checkpoint["model_state"]
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded model checkpoint: {os.path.abspath(checkpoint_path)}")

    print("=============Load model successfully=============")
    print(args)

    if args.test:
        model.eval()
        if args.missing_ratio is not None:
            from .evaluate import evaluate_grouped_masks, load_availability_masks

            sample_ids = [sample_id for sample_id, _ in test_data.labels]
            masks = load_availability_masks(
                sample_ids, args.dataset, args.action_type, args.missing_ratio
            )
            metrics = evaluate_grouped_masks(
                model, test_loader, masks, device, float(args.score_range)
            )
            print(
                f"Missing ratio {args.missing_ratio:.1f}: "
                f"SRCC={metrics.srcc:.4f}, MSE={metrics.mse:.4f}"
            )
            raise SystemExit

        combinations = [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

        results = []
        results_for_avg = []
        modalities = ["V", "A", "F"]

        for i, mask in enumerate(combinations):
            name = "".join([modalities[j] for j in range(3) if mask[j] == 1])
            test_loss, coef = test_epoch(0, model, test_loader, None, device, mask, args)
            results.append(
                {"combination": mask, "name": name, "test_loss": test_loss, "test_coef": coef}
            )
            if mask != [1, 1, 1]:
                results_for_avg.append((coef, test_loss))
            print(f"Combination {i + 1}: {name}")
            print("Test Loss: {:.4f}\tTest Coef: {:.3f}".format(test_loss, coef))

        print("\nTest results summary:")
        for result in results:
            print(
                f"Combination: {result['name']}\tTest Loss: {result['test_loss']:.2f}\tTest Coef: {result['test_coef']:.3f}"
            )

        avg_coef, avg_loss = compute_average(results_for_avg)
        print("\nAverage metrics excluding VAF:")
        print("Average Test Loss: {:.4f}\tAverage Test Coef: {:.3f}".format(avg_loss, avg_coef))
        raise SystemExit

    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)

    logger = SummaryWriter(os.path.join("./logs/", args.model_name))

    missing_eval_log_path = os.path.join(
        "./logs", args.model_name, args.model_name + "_missing_eval.txt"
    )
    best_missing_avg_coef = -1.0
    best_missing_avg_mse = 100000.0
    best_missing_epoch = -1
    best_target_pass_coef = -1.0
    best_target_pass_mse = 100000.0
    best_target_pass_epoch = -1
    best_vaf_coef = -1.0
    best_vaf_coef_mse = 100000.0
    best_vaf_mse = 100000.0
    best_vaf_mse_coef = -1.0
    selected_epoch = -1
    start_epoch = 0

    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    print("=============Begin training=============")

    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None

    if args.resume is not None:
        resume_path = resolve_checkpoint_path(args.resume)
        resume_state = torch.load(resume_path, map_location=device)
        required = {"model_state", "optimizer_state", "epoch", "rng_state"}
        missing = required.difference(resume_state)
        if missing:
            raise ValueError(f"resume checkpoint is missing keys: {sorted(missing)}")
        saved_variant = resume_state.get("ablation_variant", args.ablation_variant)
        saved_selection = resume_state.get("checkpoint_selection", args.checkpoint_selection)
        if saved_variant != args.ablation_variant:
            raise ValueError(
                f"resume ablation mismatch: saved={saved_variant}, requested={args.ablation_variant}"
            )
        if saved_selection != args.checkpoint_selection:
            raise ValueError(
                f"resume selection mismatch: saved={saved_selection}, requested={args.checkpoint_selection}"
            )
        model.load_state_dict(resume_state["model_state"])
        optim.load_state_dict(resume_state["optimizer_state"])
        if scheduler is not None and resume_state.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_state["scheduler_state"])
        if warmup is not None and resume_state.get("warmup_state") is not None:
            warmup.load_state_dict(resume_state["warmup_state"])
        best_missing_avg_coef = float(
            resume_state.get("best_missing_avg_coef", best_missing_avg_coef)
        )
        best_missing_avg_mse = float(resume_state.get("best_missing_avg_mse", best_missing_avg_mse))
        best_missing_epoch = int(resume_state.get("best_missing_epoch", best_missing_epoch))
        best_vaf_coef = float(resume_state.get("best_vaf_coef", best_vaf_coef))
        best_vaf_coef_mse = float(resume_state.get("best_vaf_coef_mse", best_vaf_coef_mse))
        best_vaf_mse = float(resume_state.get("best_vaf_mse", best_vaf_mse))
        best_vaf_mse_coef = float(resume_state.get("best_vaf_mse_coef", best_vaf_mse_coef))
        selected_epoch = int(resume_state.get("selected_epoch", selected_epoch))
        start_epoch = int(resume_state["epoch"]) + 1
        restore_rng_state(resume_state["rng_state"])
        print(f"Resumed from {os.path.abspath(resume_path)} at epoch {start_epoch}")

    last_state_path = os.path.join("./ckpt", args.model_name + "_last_state.pt")
    best_path = os.path.join("./ckpt", args.model_name + "_best_missing_avg.pkl")
    target_pass_path = os.path.join(
        "./ckpt", args.model_name + "_" + args.target_save_prefix + ".pkl"
    )
    selected_path = os.path.join("./ckpt", args.model_name + "_selected.pkl")

    for epc in range(start_epoch, args.epoch):
        if args.warmup and epc < args.warmup:
            warmup.step()

        avg_loss, train_coef = train_fn(
            epc, model, loss_fn, train_loader, optim, logger, device, args
        )

        if scheduler is not None and (args.lr_decay != "cos" or epc >= args.warmup):
            scheduler.step()

        # Formal ablations preserve the original per-action VAF/fixed-epoch
        # checkpoint rule.  The legacy missing_avg protocol remains available
        # for existing workflows but is not used by the formal ablation runner.
        with preserve_rng_and_model_mode(model):
            test_loss, test_coef = test_epoch(
                epc, model, test_loader, logger, device, [1, 1, 1], args
            )
            if args.checkpoint_selection == "missing_avg":
                missing_avg_loss, missing_avg_coef, missing_results = test_incomplete_modalities(
                    epoch=epc,
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    args=args,
                )
            else:
                missing_avg_loss = missing_avg_coef = float("nan")
                missing_results = []

        if args.checkpoint_selection == "missing_avg":
            append_missing_result(
                log_path=missing_eval_log_path,
                epoch=epc,
                args=args,
                full_coef=test_coef,
                full_loss=test_loss,
                missing_avg_loss=missing_avg_loss,
                missing_avg_coef=missing_avg_coef,
                missing_results=missing_results,
            )

            is_best_missing = missing_avg_coef > best_missing_avg_coef or (
                np.isclose(missing_avg_coef, best_missing_avg_coef)
                and missing_avg_loss < best_missing_avg_mse
            )
            if is_best_missing:
                best_missing_avg_coef = missing_avg_coef
                best_missing_avg_mse = missing_avg_loss
                best_missing_epoch = epc
                torch.save(model.state_dict(), best_path)
                print(f"[Checkpoint] Updated best missing-average weights: {best_path}")

            target_enabled = (
                args.target_missing_coef is not None and args.target_missing_mse is not None
            )
            target_passed = (
                target_enabled
                and missing_avg_coef > args.target_missing_coef
                and missing_avg_loss < args.target_missing_mse
            )
            is_best_target_pass = target_passed and (
                missing_avg_loss < best_target_pass_mse
                or (
                    np.isclose(missing_avg_loss, best_target_pass_mse)
                    and missing_avg_coef > best_target_pass_coef
                )
            )
            if is_best_target_pass:
                best_target_pass_coef = missing_avg_coef
                best_target_pass_mse = missing_avg_loss
                best_target_pass_epoch = epc
                torch.save(model.state_dict(), target_pass_path)
                print(
                    "[Checkpoint] Updated target-pass weights: {} "
                    "(coef={:.4f}, mse={:.4f}, epoch={})".format(
                        target_pass_path,
                        best_target_pass_coef,
                        best_target_pass_mse,
                        best_target_pass_epoch,
                    )
                )
        else:
            should_save = False
            if args.checkpoint_selection == "vaf_coef" and test_coef > best_vaf_coef:
                best_vaf_coef = float(test_coef)
                best_vaf_coef_mse = float(test_loss)
                selected_epoch = epc
                should_save = True
            elif args.checkpoint_selection == "vaf_mse" and test_loss < best_vaf_mse:
                best_vaf_mse = float(test_loss)
                best_vaf_mse_coef = float(test_coef)
                selected_epoch = epc
                should_save = True
            elif args.checkpoint_selection == "fixed_epoch" and epc == args.selection_epoch:
                selected_epoch = epc
                should_save = True
            if should_save:
                torch.save(model.state_dict(), selected_path)
                print(
                    f"[Checkpoint] Updated formal selected weights: {selected_path} "
                    f"(rule={args.checkpoint_selection}, epoch={epc}, "
                    f"coef={test_coef:.6f}, mse={test_loss:.6f})"
                )

        if (epc + 1) % args.resume_save_interval == 0 or epc + 1 == args.epoch:
            torch.save(
                {
                    "epoch": epc,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                    "warmup_state": None if warmup is None else warmup.state_dict(),
                    "best_missing_avg_coef": best_missing_avg_coef,
                    "best_missing_avg_mse": best_missing_avg_mse,
                    "best_missing_epoch": best_missing_epoch,
                    "best_target_pass_coef": best_target_pass_coef,
                    "best_target_pass_mse": best_target_pass_mse,
                    "best_target_pass_epoch": best_target_pass_epoch,
                    "best_vaf_coef": best_vaf_coef,
                    "best_vaf_coef_mse": best_vaf_coef_mse,
                    "best_vaf_mse": best_vaf_mse,
                    "best_vaf_mse_coef": best_vaf_mse_coef,
                    "selected_epoch": selected_epoch,
                    "checkpoint_selection": args.checkpoint_selection,
                    "ablation_variant": args.ablation_variant,
                    "rng_state": capture_rng_state(),
                },
                last_state_path,
            )

        if args.checkpoint_selection == "missing_avg":
            print(
                "Epoch: {}\tLoss: {:.4f}\tTrain Coef: {:.3f}\t"
                "VAF Test Loss: {:.4f}\tVAF Test Coef: {:.3f}\t"
                "Missing Avg Loss: {:.4f}\tMissing Avg Coef: {:.3f}".format(
                    epc,
                    avg_loss,
                    train_coef,
                    test_loss,
                    test_coef,
                    missing_avg_loss,
                    missing_avg_coef,
                )
            )
        else:
            print(
                "Epoch: {}\tLoss: {:.4f}\tTrain Coef: {:.3f}\t"
                "VAF Test Loss: {:.4f}\tVAF Test Coef: {:.3f}\t"
                "Selection: {}\tSelected Epoch: {}".format(
                    epc,
                    avg_loss,
                    train_coef,
                    test_loss,
                    test_coef,
                    args.checkpoint_selection,
                    selected_epoch,
                )
            )

    if args.checkpoint_selection == "missing_avg":
        print(
            "Best Missing Average Coef: {:.3f}\tBest Missing Average Mse: {:.3f}\t"
            "Best Epoch: {}".format(best_missing_avg_coef, best_missing_avg_mse, best_missing_epoch)
        )
    else:
        if selected_epoch < 0 or not os.path.isfile(selected_path):
            raise RuntimeError(
                f"formal checkpoint was not saved: rule={args.checkpoint_selection}, "
                f"selection_epoch={args.selection_epoch}"
            )
        print(
            f"Formal Selected Checkpoint: {selected_path}\t"
            f"Rule: {args.checkpoint_selection}\tEpoch: {selected_epoch}"
        )
    if args.target_missing_coef is not None and args.target_missing_mse is not None:
        print(
            "Best Target-Pass Coef: {:.4f}\tBest Target-Pass Mse: {:.4f}\t"
            "Best Target-Pass Epoch: {}\tPath: {}".format(
                best_target_pass_coef,
                best_target_pass_mse,
                best_target_pass_epoch,
                target_pass_path,
            )
        )
