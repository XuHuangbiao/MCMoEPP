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

from .models import loss, model
from .test import test_epoch, evaluate_grouped_masks, load_availability_masks


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
    if args.checkpoint_selection == "fixed_epoch":
        if args.selection_epoch is None:
            raise ValueError("--selection-epoch is required for fixed_epoch selection")
        if not 0 <= args.selection_epoch < args.epoch:
            raise ValueError("--selection-epoch must satisfy 0 <= selection_epoch < epoch")
    elif args.selection_epoch is not None:
        raise ValueError("--selection-epoch is only valid with fixed_epoch selection")
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

    best_vaf_coef = -1.0
    best_vaf_mse = 100000.0
    selected_epoch = -1

    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    print("=============Begin training=============")

    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None

    selected_path = os.path.join("./ckpt", args.model_name + "_selected.pkl")

    for epc in range(args.epoch):
        if args.warmup and epc < args.warmup:
            warmup.step()

        avg_loss, train_coef = train_fn(
            epc, model, loss_fn, train_loader, optim, logger, device, args
        )

        if scheduler is not None and (args.lr_decay != "cos" or epc >= args.warmup):
            scheduler.step()

        with preserve_rng_and_model_mode(model):
            test_loss, test_coef = test_epoch(
                epc, model, test_loader, logger, device, [1, 1, 1], args
            )
        should_save = False
        if args.checkpoint_selection == "vaf_coef" and test_coef > best_vaf_coef:
            best_vaf_coef = float(test_coef)
            selected_epoch = epc
            should_save = True
        elif args.checkpoint_selection == "vaf_mse" and test_loss < best_vaf_mse:
            best_vaf_mse = float(test_loss)
            selected_epoch = epc
            should_save = True
        elif args.checkpoint_selection == "fixed_epoch" and epc == args.selection_epoch:
            selected_epoch = epc
            should_save = True
        if should_save:
            torch.save(model.state_dict(), selected_path)
            print(
                f"[Checkpoint] Updated selected weights: {selected_path} "
                f"(rule={args.checkpoint_selection}, epoch={epc}, "
                f"coef={test_coef:.6f}, mse={test_loss:.6f})"
            )

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

    if selected_epoch < 0 or not os.path.isfile(selected_path):
        raise RuntimeError(
            f"selected checkpoint was not saved: rule={args.checkpoint_selection}, "
            f"selection_epoch={args.selection_epoch}"
        )
    print(
        f"Selected Checkpoint: {selected_path}\t"
        f"Rule: {args.checkpoint_selection}\tEpoch: {selected_epoch}"
    )
