"""CMU-MOSI/CMU-MOSEI feature reader with conversation-level split labels."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_float_tensor(value: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(value.astype(np.float32, copy=False))
    return torch.from_numpy(array)


def _load_feature(feature_root: Path, utterance_id: str) -> np.ndarray:
    feature_file = feature_root / f"{utterance_id}.npy"
    feature_dir = feature_root / utterance_id
    if feature_file.is_file():
        features = [np.load(feature_file).squeeze()]
    elif feature_dir.is_dir():
        features = [np.load(path).squeeze() for path in sorted(feature_dir.glob("*.npy"))]
    else:
        raise FileNotFoundError(
            f"Feature for utterance {utterance_id!r} was not found under {feature_root}."
        )

    feature = np.asarray(features).squeeze()
    if feature.size == 0:
        raise ValueError(f"Empty feature for utterance {utterance_id!r} under {feature_root}.")
    if feature.ndim == 2:
        feature = feature.mean(axis=0)
    elif feature.ndim > 2:
        feature = feature.reshape(feature.shape[0], -1).mean(axis=0)
    return feature.astype(np.float32, copy=False)


def _canonical_dataset_name(dataset: str) -> str:
    return {"MOSI": "CMUMOSI", "MOSEI": "CMUMOSEI"}[dataset]


class CMUSentimentDataset(Dataset):
    """One variable-length conversation per item.

    Returned modality order is ``[text, visual, audio]``. ``collate_fn`` pads
    conversations and returns an utterance mask so padded labels never affect
    the loss or metrics.
    """

    def __init__(self, split: str, args) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split: {split!r}")
        self.split = split
        self.dataset_name = _canonical_dataset_name(args.dataset)

        label_path = Path(args.label_path).expanduser().resolve()
        if not label_path.is_file():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        self.feature_roots = {
            "text": Path(args.text_path).expanduser().resolve(),
            "visual": Path(args.video_path).expanduser().resolve(),
            "audio": Path(args.audio_path).expanduser().resolve(),
        }
        for root in self.feature_roots.values():
            if not root.exists():
                raise FileNotFoundError(f"Feature directory not found: {root}")

        with label_path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        (
            self.video_ids,
            self.video_labels,
            _video_speakers,
            _video_sentences,
            train_ids,
            valid_ids,
            test_ids,
        ) = payload
        split_ids = {"train": train_ids, "valid": valid_ids, "test": test_ids}
        self.conversation_ids = sorted(split_ids[split])
        if not self.conversation_ids:
            raise ValueError(f"No conversations in split {split!r}: {label_path}")

        self.preload = os.environ.get("MCMOE_PLUS_PRELOAD", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self.cache: Dict[str, Dict[str, np.ndarray]] = {
            "text": {},
            "visual": {},
            "audio": {},
        }
        first_id = self.video_ids[self.conversation_ids[0]][0]
        args.text_dim = int(self._feature("text", first_id).shape[-1])
        args.visual_dim = int(self._feature("visual", first_id).shape[-1])
        args.audio_dim = int(self._feature("audio", first_id).shape[-1])
        args.label_min = -3.0
        args.label_max = 3.0
        args.score_range = 6.0

        if self.preload:
            for conversation_id in self.conversation_ids:
                for utterance_id in self.video_ids[conversation_id]:
                    for modality in self.cache:
                        self._feature(modality, utterance_id)

        print(
            f"Loaded {self.dataset_name} split={split}: conversations={len(self)}, "
            f"dims={args.text_dim}/{args.visual_dim}/{args.audio_dim}",
            flush=True,
        )

    def _feature(self, modality: str, utterance_id: str) -> np.ndarray:
        cached = self.cache[modality]
        if utterance_id not in cached:
            value = _load_feature(self.feature_roots[modality], utterance_id)
            if self.preload:
                cached[utterance_id] = value
            return value
        return cached[utterance_id]

    def __len__(self) -> int:
        return len(self.conversation_ids)

    def __getitem__(self, index: int):
        conversation_id = self.conversation_ids[index]
        utterance_ids = self.video_ids[conversation_id]
        labels = self.video_labels[conversation_id]

        text = np.asarray([self._feature("text", uid) for uid in utterance_ids])
        visual = np.asarray([self._feature("visual", uid) for uid in utterance_ids])
        audio = np.asarray([self._feature("audio", uid) for uid in utterance_ids])
        normalized_labels = np.asarray(
            [(float(label) + 3.0) / 6.0 for label in labels], dtype=np.float32
        )
        return tuple(map(_to_float_tensor, (text, visual, audio, normalized_labels)))

    @staticmethod
    def collate_fn(batch):
        text, visual, audio, labels = zip(*batch)
        batch_size = len(batch)
        max_length = max(item.size(0) for item in labels)

        def pad_features(items):
            output = items[0].new_zeros((batch_size, max_length, items[0].size(-1)))
            for index, item in enumerate(items):
                output[index, : item.size(0)] = item
            return output

        padded_labels = labels[0].new_zeros((batch_size, max_length))
        utterance_mask = labels[0].new_zeros((batch_size, max_length))
        for index, item in enumerate(labels):
            padded_labels[index, : item.size(0)] = item
            utterance_mask[index, : item.size(0)] = 1.0
        return (
            pad_features(text),
            pad_features(visual),
            pad_features(audio),
            padded_labels,
            utterance_mask,
        )
