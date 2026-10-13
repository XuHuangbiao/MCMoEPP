import math

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import Transformer


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=136):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[: x.size(0), :].detach()
        x = x + pe
        return self.dropout(x)


def generate_modality_mask():
    combinations = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    idx = torch.randint(0, 6, [1])
    return combinations[idx.item()]


class AGMG(nn.Module):
    def __init__(self, hidden_dim, recon_layers=2):
        super().__init__()
        if recon_layers < 1:
            raise ValueError("recon_layers must be positive")
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=4,
                    batch_first=True,
                    kdim=hidden_dim,
                    vdim=hidden_dim,
                )
                for _ in range(recon_layers)
            ]
        )
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, exist_feats, mask_feat):
        context = torch.cat(exist_feats, dim=1)
        for attn in self.cross_attn:
            mask_feat, _ = attn(mask_feat, context, context)
        gate = self.gate(torch.cat([mask_feat.mean(1), context.mean(1)], -1))
        return gate.unsqueeze(1) * mask_feat


class MoE_AQA(nn.Module):
    ABLATION_VARIANTS = (
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
        "no_coord",
        "no_lrecon",
        "no_lalign",
        "no_ldiv",
    )

    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.set_ablation_variant(getattr(config, "ablation_variant", "full"))
        self.in_proj = nn.ModuleDict(
            {
                "v": self._build_proj(in_dim, hidden_dim),
                "a": self._build_proj(768, hidden_dim),
                "f": self._build_proj(1024, hidden_dim),
            }
        )
        self.experts = nn.ModuleDict(
            {
                "v": self._build_expert(hidden_dim, hidden_dim),
                "a": self._build_expert(hidden_dim, hidden_dim),
                "f": self._build_expert(hidden_dim, hidden_dim),
                "s": self._build_expert(hidden_dim, hidden_dim),
            }
        )
        self.generators = nn.ModuleDict(
            {
                "v": AGMG(hidden_dim, getattr(config, "recon_layers", 2)),
                "a": AGMG(hidden_dim, getattr(config, "recon_layers", 2)),
                "f": AGMG(hidden_dim, getattr(config, "recon_layers", 2)),
            }
        )
        router_outputs = 3 if self.ablation_variant in {"no_anchor_residual", "no_ma_moe"} else 2
        router_inputs = hidden_dim if self.ablation_variant == "no_ma_moe" else 3 * hidden_dim
        self.router_mix = Mlp(router_inputs, hidden_dim // 2, router_outputs)
        self.stream_embed = nn.Embedding(3, hidden_dim)
        self.mask_embed = nn.Linear(3, hidden_dim)
        self.stream_reliability = Mlp(3 * hidden_dim, hidden_dim // 2, 1)
        # Start close to one so specific streams are not suppressed early in training.
        nn.init.constant_(self.stream_reliability.fc2.bias, 4.0)
        self.shared_specific_gate = Mlp(3 * hidden_dim, hidden_dim // 2, 1)

        # Learned cap priors for the eight visibility patterns (v*4 + a*2 + f).
        self.gen_cap_prior = nn.Embedding(8, 1)
        self.shared_cap_prior = nn.Embedding(8, 1)
        self._init_cap_priors()

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(3 * hidden_dim, 2 * hidden_dim, kernel_size=1),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(True),
            nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        # Query-score weights move with the model without entering its state dict.
        self.register_buffer("weight", torch.linspace(0, 1, n_query), persistent=False)
        self.pos_encoder = PositionalEncoding(hidden_dim)

    def _init_cap_priors(self):
        """Initialize gen_cap_prior and shared_cap_prior so that
        sigmoid(prior[mask_id]) == n_vis / 3 for every mask_id.

        mask_id = v * 4 + a * 2 + f, n_vis = popcount(mask_id).
        The prior is initialized from the number of available modalities.
        """
        with torch.no_grad():

            def to_logit(p):
                p = max(min(float(p), 1.0 - 1e-4), 1e-4)
                return math.log(p / (1.0 - p))

            for mid in range(8):
                v = (mid >> 2) & 1
                a = (mid >> 1) & 1
                f = mid & 1
                # mask_id == 0 should not appear at inference; use n_vis = 1 as a
                # conservative fallback so sigmoid(logit(1/3)) == 1/3.
                n_vis = max(v + a + f, 1)
                init_logit = to_logit(n_vis / 3.0)
                self.gen_cap_prior.weight[mid, 0] = init_logit
                self.shared_cap_prior.weight[mid, 0] = init_logit

    def _build_expert(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim // 2),
            nn.GELU(),
            nn.Conv1d(out_dim // 2, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
        )

    def _build_proj(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=1),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(in_dim // 2, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
        )

    def _expand_mask_tensor(self, mask_tensor, b):
        if mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).repeat(b, 1)
        elif mask_tensor.size(0) == 1:
            mask_tensor = mask_tensor.repeat(b, 1)
        return mask_tensor

    def _build_router_input(self, feat, mask_tensor, stream_idx):
        b, t, _ = feat.shape
        mask_tensor = self._expand_mask_tensor(mask_tensor, b)
        mask_emb = self.mask_embed(mask_tensor.float()).unsqueeze(1).repeat(1, t, 1)
        stream_ids = torch.full((b, t), stream_idx, dtype=torch.long, device=feat.device)
        stream_emb = self.stream_embed(stream_ids)
        return torch.cat([feat, stream_emb, mask_emb], dim=-1)

    def _conditioned_router(self, feat, mask_tensor, stream_idx):
        logits = self.router_mix(self._build_router_input(feat, mask_tensor, stream_idx))
        if self.ablation_variant == "no_anchor_residual":
            return torch.softmax(logits, dim=-1)
        return torch.sigmoid(logits)

    def _feature_only_router(self, feat):
        """Feature-only router for the trained no_ma_router ablation."""
        zeros = torch.zeros_like(feat)
        return torch.sigmoid(self.router_mix(torch.cat([feat, zeros, zeros], dim=-1)))

    def _plain_three_expert_router(self, feat):
        """Ordinary feature-only three-expert router used by w/o MA-MoE."""
        return torch.softmax(self.router_mix(feat), dim=-1)

    def _complementary_specific_fusion(self, e0, e1, e2, router_weight, anchor_idx):
        if anchor_idx == 0:
            anchor = e0
            w1 = router_weight[:, :, 0:1]
            w2 = router_weight[:, :, 1:2]
            residual = w1 * (e1 - anchor) + w2 * (e2 - anchor)
            denom = 1.0 + w1 + w2
        elif anchor_idx == 1:
            anchor = e1
            w0 = router_weight[:, :, 0:1]
            w2 = router_weight[:, :, 1:2]
            residual = w0 * (e0 - anchor) + w2 * (e2 - anchor)
            denom = 1.0 + w0 + w2
        else:
            anchor = e2
            w0 = router_weight[:, :, 0:1]
            w1 = router_weight[:, :, 1:2]
            residual = w0 * (e0 - anchor) + w1 * (e1 - anchor)
            denom = 1.0 + w0 + w1
        fusion = anchor + residual / denom
        return fusion

    def set_ablation_variant(self, variant):
        if variant not in self.ABLATION_VARIANTS:
            raise ValueError(
                f"unknown ablation_variant={variant!r}; expected one of {self.ABLATION_VARIANTS}"
            )
        self.ablation_variant = variant

    @staticmethod
    def _softmax_specific_fusion(e0, e1, e2, weights):
        return weights[..., 0:1] * e0 + weights[..., 1:2] * e1 + weights[..., 2:3] * e2

    def _build_shared_feat(self, v_mask, a_mask, f_mask):
        shared_base = (v_mask + a_mask + f_mask) / 3.0
        shared_feat = self.experts["s"](shared_base.transpose(1, 2)).transpose(1, 2)
        return shared_feat

    def _get_mask_id(self, mask_tensor):
        """Map a (b, 3) binary visibility mask to a (b,) long mask_id in [0, 7].
        mask_id = v * 4 + a * 2 + f.
        """
        return (
            mask_tensor[:, 0].long() * 4 + mask_tensor[:, 1].long() * 2 + mask_tensor[:, 2].long()
        ).clamp(0, 7)

    def _stream_reliability_gate(self, feat, mask_tensor, stream_idx):
        b, t, c = feat.shape
        mask_tensor = self._expand_mask_tensor(mask_tensor, b)
        gate_input = self._build_router_input(feat, mask_tensor, stream_idx)
        raw_rel = torch.sigmoid(self.stream_reliability(gate_input))
        real_flag = mask_tensor[:, stream_idx : stream_idx + 1].view(b, 1, 1)
        # Learnable mask-conditioned cap. Initialized to n_vis / 3, then learnable.
        mask_id = self._get_mask_id(mask_tensor)
        gen_cap = torch.sigmoid(self.gen_cap_prior(mask_id)).view(b, 1, 1)
        reliability = real_flag * raw_rel + (1.0 - real_flag) * raw_rel * gen_cap
        return reliability

    def _global_shared_specific_fusion(self, specific_global, shared_feat, mask_tensor):
        b, t, c = specific_global.shape
        mask_tensor = self._expand_mask_tensor(mask_tensor, b)
        mask_emb = self.mask_embed(mask_tensor.float()).unsqueeze(1).repeat(1, t, 1)
        gate_input = torch.cat([specific_global, shared_feat, mask_emb], dim=-1)
        raw_shared_ratio = torch.sigmoid(self.shared_specific_gate(gate_input))
        # Learnable mask-conditioned cap. Initialized to n_vis / 3, then learnable.
        mask_id = self._get_mask_id(mask_tensor)
        shared_cap = torch.sigmoid(self.shared_cap_prior(mask_id)).view(b, 1, 1)
        shared_ratio = raw_shared_ratio * shared_cap
        fusion = specific_global + shared_ratio * (shared_feat - specific_global)
        return fusion, shared_ratio

    def forward(self, video, audio, flow, mask):
        if self.training:
            return self.forward_train(video, audio, flow, mask)
        else:
            return self.forward_test(video, audio, flow, mask)

    def forward_train(self, video, audio, flow, mask):
        variant = self.ablation_variant
        b, t, c = video.shape
        v = self.in_proj["v"](video.transpose(1, 2)).transpose(1, 2)
        a = self.in_proj["a"](audio.transpose(1, 2)).transpose(1, 2)
        f = self.in_proj["f"](flow.transpose(1, 2)).transpose(1, 2)
        use_stem = variant not in {"baseline", "no_stem"}
        proj_v = self.transformer.encoder(v) if use_stem else v
        proj_a = self.transformer.encoder(a) if use_stem else a
        proj_f = self.transformer.encoder(f) if use_stem else f
        train_mask = generate_modality_mask()
        mask_tensor = torch.tensor(train_mask, device=video.device).float().view(1, 3)
        v_m, a_m, f_m = train_mask
        v_mask = proj_v if v_m == 1 else torch.zeros_like(proj_v)
        a_mask = proj_a if a_m == 1 else torch.zeros_like(proj_a)
        f_mask = proj_f if f_m == 1 else torch.zeros_like(proj_f)
        missing = [key for key, visible in zip(("v", "a", "f"), (v_m, a_m, f_m)) if visible == 0]
        recon_loss = video.new_zeros(())
        if variant not in {"baseline", "stem", "no_agmg"}:
            for m in missing:
                exist_feats = []
                if "v" not in missing:
                    exist_feats.append(v_mask)
                if "a" not in missing:
                    exist_feats.append(a_mask)
                if "f" not in missing:
                    exist_feats.append(f_mask)
                if m == "v":
                    v_mask = self.generators["v"](exist_feats, torch.zeros_like(v_mask))
                    if variant != "no_lrecon":
                        recon_loss = recon_loss + F.mse_loss(v_mask, proj_v)
                elif m == "a":
                    a_mask = self.generators["a"](exist_feats, torch.zeros_like(a_mask))
                    if variant != "no_lrecon":
                        recon_loss = recon_loss + F.mse_loss(a_mask, proj_a)
                else:
                    f_mask = self.generators["f"](exist_feats, torch.zeros_like(f_mask))
                    if variant != "no_lrecon":
                        recon_loss = recon_loss + F.mse_loss(f_mask, proj_f)

        if variant in {"baseline", "stem", "agmg"}:
            specific_v, specific_a, specific_f = v_mask, a_mask, f_mask
            specific_cat = torch.cat([specific_v, specific_a, specific_f], -1)
            specific_global = self.fusion(specific_cat.transpose(1, 2)).transpose(1, 2)
            fusion = specific_global
        else:
            expert_vv = self.experts["v"](v_mask.transpose(1, 2)).transpose(1, 2)
            expert_va = self.experts["a"](v_mask.transpose(1, 2)).transpose(1, 2)
            expert_vf = self.experts["f"](v_mask.transpose(1, 2)).transpose(1, 2)
            expert_av = self.experts["v"](a_mask.transpose(1, 2)).transpose(1, 2)
            expert_aa = self.experts["a"](a_mask.transpose(1, 2)).transpose(1, 2)
            expert_af = self.experts["f"](a_mask.transpose(1, 2)).transpose(1, 2)
            expert_fv = self.experts["v"](f_mask.transpose(1, 2)).transpose(1, 2)
            expert_fa = self.experts["a"](f_mask.transpose(1, 2)).transpose(1, 2)
            expert_ff = self.experts["f"](f_mask.transpose(1, 2)).transpose(1, 2)
            if variant == "no_ma_moe":
                router_weight_v = self._plain_three_expert_router(v_mask)
                router_weight_a = self._plain_three_expert_router(a_mask)
                router_weight_f = self._plain_three_expert_router(f_mask)
            elif variant == "no_ma_router":
                router_weight_v = self._feature_only_router(v_mask)
                router_weight_a = self._feature_only_router(a_mask)
                router_weight_f = self._feature_only_router(f_mask)
            else:
                router_weight_v = self._conditioned_router(v_mask, mask_tensor, stream_idx=0)
                router_weight_a = self._conditioned_router(a_mask, mask_tensor, stream_idx=1)
                router_weight_f = self._conditioned_router(f_mask, mask_tensor, stream_idx=2)

            if variant in {"no_anchor_residual", "no_ma_moe"}:
                specific_v = self._softmax_specific_fusion(
                    expert_vv, expert_va, expert_vf, router_weight_v
                )
                specific_a = self._softmax_specific_fusion(
                    expert_av, expert_aa, expert_af, router_weight_a
                )
                specific_f = self._softmax_specific_fusion(
                    expert_fv, expert_fa, expert_ff, router_weight_f
                )
            else:
                specific_v = self._complementary_specific_fusion(
                    expert_vv, expert_va, expert_vf, router_weight_v, anchor_idx=0
                )
                specific_a = self._complementary_specific_fusion(
                    expert_av, expert_aa, expert_af, router_weight_a, anchor_idx=1
                )
                specific_f = self._complementary_specific_fusion(
                    expert_fv, expert_fa, expert_ff, router_weight_f, anchor_idx=2
                )

            specific_v_pre, specific_a_pre, specific_f_pre = specific_v, specific_a, specific_f
            if variant in {"ma_moe", "no_coord"}:
                rel_v = torch.ones_like(specific_v[..., :1])
                rel_a = torch.ones_like(specific_a[..., :1])
                rel_f = torch.ones_like(specific_f[..., :1])
            else:
                rel_v = self._stream_reliability_gate(specific_v, mask_tensor, stream_idx=0)
                rel_a = self._stream_reliability_gate(specific_a, mask_tensor, stream_idx=1)
                rel_f = self._stream_reliability_gate(specific_f, mask_tensor, stream_idx=2)
            specific_v = rel_v * specific_v
            specific_a = rel_a * specific_a
            specific_f = rel_f * specific_f
            specific_cat = torch.cat([specific_v, specific_a, specific_f], -1)
            specific_global = self.fusion(specific_cat.transpose(1, 2)).transpose(1, 2)

            if variant == "ma_moe":
                fusion = specific_global
            elif variant == "no_shared_expert":
                shared_feat = (v_mask + a_mask + f_mask) / 3.0
            else:
                shared_feat = self._build_shared_feat(v_mask, a_mask, f_mask)
            if variant != "ma_moe":
                if variant == "no_coord":
                    shared_ratio = torch.full_like(specific_global[..., :1], 0.5)
                    fusion = specific_global + shared_ratio * (shared_feat - specific_global)
                else:
                    fusion, shared_ratio = self._global_shared_specific_fusion(
                        specific_global, shared_feat, mask_tensor
                    )

            if variant != "no_lalign":
                expert_v = self.experts["v"](proj_v.transpose(1, 2)).transpose(1, 2)
                expert_a = self.experts["a"](proj_a.transpose(1, 2)).transpose(1, 2)
                expert_f = self.experts["f"](proj_f.transpose(1, 2)).transpose(1, 2)
                align_v, align_a, align_f = specific_v_pre, specific_a_pre, specific_f_pre
                recon_loss = recon_loss + F.kl_div(
                    torch.log_softmax(align_v, -1), torch.softmax(expert_v, -1), reduction="mean"
                )
                recon_loss = recon_loss + F.kl_div(
                    torch.log_softmax(align_a, -1), torch.softmax(expert_a, -1), reduction="mean"
                )
                recon_loss = recon_loss + F.kl_div(
                    torch.log_softmax(align_f, -1), torch.softmax(expert_f, -1), reduction="mean"
                )

        prototype = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        prototype = self.pos_encoder(prototype)
        fea, _ = self.transformer.decoder(prototype, fusion)
        s = self.regressor(fea)
        s = torch.diagonal(s, dim1=-2, dim2=-1)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return {
            "output": out,
            "embed": fea,
            "embed2": None,
            "recon_loss": recon_loss,
        }

    def forward_test(self, video, audio, flow, mask):
        variant = self.ablation_variant
        b, t, c = video.shape
        mask_tensor = torch.tensor(mask, device=video.device).float().view(1, 3)
        v_visible, a_visible, f_visible = mask
        use_stem = variant not in {"baseline", "no_stem"}
        if v_visible == 1:
            v = self.in_proj["v"](video.transpose(1, 2)).transpose(1, 2)
            v_mask = self.transformer.encoder(v) if use_stem else v
        else:
            v_mask = torch.zeros(b, t, self.hidden_dim).to(video.device)
        if a_visible == 1:
            a = self.in_proj["a"](audio.transpose(1, 2)).transpose(1, 2)
            a_mask = self.transformer.encoder(a) if use_stem else a
        else:
            a_mask = torch.zeros(b, t, self.hidden_dim).to(audio.device)
        if f_visible == 1:
            f = self.in_proj["f"](flow.transpose(1, 2)).transpose(1, 2)
            f_mask = self.transformer.encoder(f) if use_stem else f
        else:
            f_mask = torch.zeros(b, t, self.hidden_dim).to(flow.device)

        missing = [
            key
            for key, visible in zip(("v", "a", "f"), (v_visible, a_visible, f_visible))
            if visible == 0
        ]
        if variant not in {"baseline", "stem", "no_agmg"}:
            for m in missing:
                exist_feats = []
                if "v" not in missing:
                    exist_feats.append(v_mask)
                if "a" not in missing:
                    exist_feats.append(a_mask)
                if "f" not in missing:
                    exist_feats.append(f_mask)
                if m == "v":
                    v_mask = self.generators["v"](exist_feats, torch.zeros_like(v_mask))
                elif m == "a":
                    a_mask = self.generators["a"](exist_feats, torch.zeros_like(a_mask))
                else:
                    f_mask = self.generators["f"](exist_feats, torch.zeros_like(f_mask))

        if variant in {"baseline", "stem", "agmg"}:
            # Progressive mechanism-only rows before MA-MoE: the three current
            # streams enter the original CFM directly. No expert/router/MASEC
            # parameter is called.
            specific_v, specific_a, specific_f = v_mask, a_mask, f_mask
            specific_cat = torch.cat([specific_v, specific_a, specific_f], -1)
            specific_global = self.fusion(specific_cat.transpose(1, 2)).transpose(1, 2)
            fusion = specific_global
        else:
            expert_vv = self.experts["v"](v_mask.transpose(1, 2)).transpose(1, 2)
            expert_va = self.experts["a"](v_mask.transpose(1, 2)).transpose(1, 2)
            expert_vf = self.experts["f"](v_mask.transpose(1, 2)).transpose(1, 2)
            expert_av = self.experts["v"](a_mask.transpose(1, 2)).transpose(1, 2)
            expert_aa = self.experts["a"](a_mask.transpose(1, 2)).transpose(1, 2)
            expert_af = self.experts["f"](a_mask.transpose(1, 2)).transpose(1, 2)
            expert_fv = self.experts["v"](f_mask.transpose(1, 2)).transpose(1, 2)
            expert_fa = self.experts["a"](f_mask.transpose(1, 2)).transpose(1, 2)
            expert_ff = self.experts["f"](f_mask.transpose(1, 2)).transpose(1, 2)
            if variant == "no_ma_moe":
                router_weight_v = self._plain_three_expert_router(v_mask)
                router_weight_a = self._plain_three_expert_router(a_mask)
                router_weight_f = self._plain_three_expert_router(f_mask)
            elif variant == "no_ma_router":
                router_weight_v = self._feature_only_router(v_mask)
                router_weight_a = self._feature_only_router(a_mask)
                router_weight_f = self._feature_only_router(f_mask)
            else:
                router_weight_v = self._conditioned_router(v_mask, mask_tensor, stream_idx=0)
                router_weight_a = self._conditioned_router(a_mask, mask_tensor, stream_idx=1)
                router_weight_f = self._conditioned_router(f_mask, mask_tensor, stream_idx=2)

            if variant == "no_ma_moe":
                specific_v = self._softmax_specific_fusion(
                    expert_vv, expert_va, expert_vf, router_weight_v
                )
                specific_a = self._softmax_specific_fusion(
                    expert_av, expert_aa, expert_af, router_weight_a
                )
                specific_f = self._softmax_specific_fusion(
                    expert_fv, expert_fa, expert_ff, router_weight_f
                )
            elif variant == "no_anchor_residual":
                specific_v = self._softmax_specific_fusion(
                    expert_vv, expert_va, expert_vf, router_weight_v
                )
                specific_a = self._softmax_specific_fusion(
                    expert_av, expert_aa, expert_af, router_weight_a
                )
                specific_f = self._softmax_specific_fusion(
                    expert_fv, expert_fa, expert_ff, router_weight_f
                )
            else:
                specific_v = self._complementary_specific_fusion(
                    expert_vv, expert_va, expert_vf, router_weight_v, anchor_idx=0
                )
                specific_a = self._complementary_specific_fusion(
                    expert_av, expert_aa, expert_af, router_weight_a, anchor_idx=1
                )
                specific_f = self._complementary_specific_fusion(
                    expert_fv, expert_fa, expert_ff, router_weight_f, anchor_idx=2
                )

            if variant in {"ma_moe", "no_coord"}:
                rel_v = torch.ones_like(specific_v[..., :1])
                rel_a = torch.ones_like(specific_a[..., :1])
                rel_f = torch.ones_like(specific_f[..., :1])
            else:
                rel_v = self._stream_reliability_gate(specific_v, mask_tensor, stream_idx=0)
                rel_a = self._stream_reliability_gate(specific_a, mask_tensor, stream_idx=1)
                rel_f = self._stream_reliability_gate(specific_f, mask_tensor, stream_idx=2)
            specific_v = rel_v * specific_v
            specific_a = rel_a * specific_a
            specific_f = rel_f * specific_f
            specific_cat = torch.cat([specific_v, specific_a, specific_f], -1)
            specific_global = self.fusion(specific_cat.transpose(1, 2)).transpose(1, 2)

            if variant == "ma_moe":
                fusion = specific_global
            else:
                if variant == "no_shared_expert":
                    shared_feat = (v_mask + a_mask + f_mask) / 3.0
                else:
                    shared_feat = self._build_shared_feat(v_mask, a_mask, f_mask)
                if variant == "no_coord":
                    shared_ratio = torch.full_like(specific_global[..., :1], 0.5)
                    fusion = specific_global + shared_ratio * (shared_feat - specific_global)
                else:
                    fusion, shared_ratio = self._global_shared_specific_fusion(
                        specific_global, shared_feat, mask_tensor
                    )
        prototype = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        prototype = self.pos_encoder(prototype)
        fea, _ = self.transformer.decoder(prototype, fusion)
        s = self.regressor(fea)
        s = torch.diagonal(s, dim1=-2, dim2=-1)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return {"output": out}
