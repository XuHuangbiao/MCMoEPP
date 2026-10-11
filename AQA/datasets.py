import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _crop_or_pad_triplet(video_feat, audio_feat, flow_feat, clip_num, train=True):
    """Crop or pad three aligned modality feature sequences to the same clip length."""
    seq_len = len(video_feat)

    if seq_len > clip_num:
        max_st = seq_len - clip_num
        st = np.random.randint(0, max_st + 1) if train else max_st // 2
        video_feat = video_feat[st : st + clip_num]
        audio_feat = audio_feat[st : st + clip_num]
        flow_feat = flow_feat[st : st + clip_num]

    elif seq_len < clip_num:

        def _pad(x):
            out = np.zeros((clip_num, x.shape[1]), dtype=x.dtype)
            out[: x.shape[0]] = x
            return out

        video_feat = _pad(video_feat)
        audio_feat = _pad(audio_feat)
        flow_feat = _pad(flow_feat)

    return video_feat, audio_feat, flow_feat


def _to_float_tensor(x):
    x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    return torch.from_numpy(x)


class _BaseAQADataset(Dataset):
    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range

    def _finalize_triplet(self, video_feat, audio_feat, flow_feat):
        video_feat, audio_feat, flow_feat = _crop_or_pad_triplet(
            video_feat, audio_feat, flow_feat, clip_num=self.clip_num, train=self.train
        )
        return (
            _to_float_tensor(video_feat),
            _to_float_tensor(audio_feat),
            _to_float_tensor(flow_feat),
        )


class FS1000Dataset(_BaseAQADataset):
    """
    FS1000 dataset, rewritten in the same style as FisVDataset:
    1. inherit _BaseAQADataset;
    2. parse labels in read_label;
    3. preload features in __init__;
    4. use _finalize_triplet for crop/pad/tensor conversion.

    Note:
    FS1000 stores each sample as a separate .npy file rather than a single dict .npy.
    Therefore, this class preloads all sample files into dictionaries in __init__.
    """

    def __init__(
        self,
        video_feat_path,
        audio_feat_path,
        flow_feat_path,
        label_path,
        clip_num=26,
        action_type="TES",
        train=True,
        args=None,
    ):
        self.train = train
        self.video_path = video_feat_path
        self.audio_path = audio_feat_path
        self.flow_path = flow_feat_path
        self.erase_path = video_feat_path + "_erTrue"

        score_idx = {"TES": 130, "PCS": 60, "SS": 10, "TR": 10, "PE": 10, "CO": 10, "IN": 10}
        self.score_range = float(score_idx[action_type])
        if args is not None:
            args.score_range = self.score_range

        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

        print("Loading FS1000 feature files...", flush=True)
        self.video_dict = {}
        self.audio_dict = {}
        self.flow_dict = {}

        for key, _ in self.labels:
            video_feat = np.load(os.path.join(self.video_path, key + ".npy"))
            audio_feat = np.load(os.path.join(self.audio_path, key + ".npy"))
            flow_feat = np.load(os.path.join(self.flow_path, key + ".npy"))

            # Keep the original FS1000 processing: average video feature over axis 1.
            video_feat = video_feat.mean(1)

            self.video_dict[key] = video_feat
            self.audio_dict[key] = audio_feat
            self.flow_dict[key] = flow_feat

        print("FS1000 feature files loaded.", flush=True)

    def read_label(self, label_path, action_type):
        idx = {"TES": 1, "PCS": 2, "SS": 3, "TR": 4, "PE": 5, "CO": 6, "IN": 7}
        labels = []
        scores = []

        with open(label_path, "r") as fr:
            for line in fr:
                line = line.strip().split()
                s = float(line[idx[action_type]])
                if action_type == "PCS":
                    s = s / float(line[8])
                labels.append([line[0], s])
                scores.append(s)

        print("max:", max(scores))
        return labels

    def __getitem__(self, idx):
        key, score = self.labels[idx]
        video_feat = self.video_dict[key]
        audio_feat = self.audio_dict[key]
        flow_feat = self.flow_dict[key]

        video_feat, audio_feat, flow_feat = self._finalize_triplet(
            video_feat, audio_feat, flow_feat
        )
        return video_feat, audio_feat, flow_feat, self.normalize_score(score)


class RGDataset(_BaseAQADataset):
    """
    RG dataset, rewritten in the same style as FisVDataset:
    1. inherit _BaseAQADataset;
    2. parse labels in read_label;
    3. preload dict feature banks in __init__;
    4. use _finalize_triplet for crop/pad/tensor conversion.
    """

    def __init__(
        self,
        video_feat_path,
        audio_feat_path,
        flow_feat_path,
        label_path,
        clip_num=26,
        action_type="Ball",
        train=True,
        args=None,
    ):
        self.train = train
        self.video_path = os.path.join(video_feat_path, action_type + "_rgb_VST.npy")
        self.audio_path = os.path.join(audio_feat_path, action_type + "_audio_AST.npy")
        self.flow_path = os.path.join(flow_feat_path, action_type + "_flow_I3D.npy")
        self.erase_path = video_feat_path + "_erTrue"

        self.score_range = 25.0
        if args is not None:
            args.score_range = self.score_range

        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

        print("Loading RG feature dicts...", flush=True)
        self.video_dict = np.load(self.video_path, allow_pickle=True).item()
        self.audio_dict = np.load(self.audio_path, allow_pickle=True).item()
        self.flow_dict = np.load(self.flow_path, allow_pickle=True).item()
        print("RG feature dicts loaded.", flush=True)

    def read_label(self, label_path, action_type):
        idx = {"Difficulty_Score": 1, "Execution_Score": 2, "Total_Score": 3}
        labels = []

        with open(label_path, "r") as fr:
            for i, line in enumerate(fr):
                if i == 0:
                    continue
                line = line.strip().split()
                if action_type == "all" or action_type == line[0].split("_")[0]:
                    labels.append([line[0], float(line[idx["Total_Score"]])])

        return labels

    def __getitem__(self, idx):
        key, score = self.labels[idx]
        video_feat = self.video_dict[key]
        audio_feat = self.audio_dict[key]
        flow_feat = self.flow_dict[key]

        video_feat, audio_feat, flow_feat = self._finalize_triplet(
            video_feat, audio_feat, flow_feat
        )
        return video_feat, audio_feat, flow_feat, self.normalize_score(score)


class FisVDataset(_BaseAQADataset):
    """
    FisV dataset.
    This keeps the original FisV-style structure and is used as the template for
    FS1000Dataset and RGDataset.
    """

    def __init__(
        self,
        video_feat_path,
        audio_feat_path,
        flow_feat_path,
        label_path,
        clip_num=26,
        action_type="TES",
        train=True,
        args=None,
    ):
        self.train = train
        self.video_path = os.path.join(video_feat_path, "FISV_rgb_VST.npy")
        self.audio_path = os.path.join(audio_feat_path, "FISV_audio_AST.npy")
        self.flow_path = os.path.join(flow_feat_path, "FISV_flow_I3D.npy")

        if action_type == "TES":
            self.score_range = 45.0
        else:
            self.score_range = 40.0

        if args is not None:
            args.score_range = self.score_range

        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)

        print("Loading FisV feature dicts...", flush=True)
        self.video_dict = np.load(self.video_path, allow_pickle=True).item()
        self.audio_dict = np.load(self.audio_path, allow_pickle=True).item()
        self.flow_dict = np.load(self.flow_path, allow_pickle=True).item()
        print("FisV feature dicts loaded.", flush=True)

    def read_label(self, label_path, action_type):
        idx = {"TES": 1, "PCS": 2}
        labels = []

        with open(label_path, "r") as fr:
            for i, line in enumerate(fr):
                if i == 0:
                    continue
                line = line.strip().split()
                labels.append([line[0], float(line[idx[action_type]])])

        return labels

    def __getitem__(self, idx):
        key, score = self.labels[idx]
        video_feat = self.video_dict[key]
        audio_feat = self.audio_dict[key]
        flow_feat = self.flow_dict[key]

        video_feat, audio_feat, flow_feat = self._finalize_triplet(
            video_feat, audio_feat, flow_feat
        )
        return video_feat, audio_feat, flow_feat, self.normalize_score(score)
