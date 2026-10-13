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


class AGMG(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=4,
                    batch_first=True,
                    kdim=hidden_dim,
                    vdim=hidden_dim,
                )
                for _ in range(2)
            ]
        )
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, exist_feats, mask_feat):
        context = torch.cat(exist_feats, dim=1)
        for attn in self.cross_attn:
            mask_feat, _ = attn(mask_feat, context, context)
        gate = self.gate(torch.cat([mask_feat.mean(1), context.mean(1)], -1))
        return gate.unsqueeze(1) * mask_feat


class MoE_Sentiment(nn.Module):
    UNIFIED_VARIANTS = {
        "mcmoe_pp",
        "ma_moe",
        "wo_ma_router",
        "wo_anchor_residual",
        "wo_shared_expert",
        "wo_coord",
        "wo_gamma_stream",
        "wo_beta_global",
    }

    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ablation_variant = getattr(config, "model_variant", "mcmoe_pp")
        if self.ablation_variant not in self.UNIFIED_VARIANTS:
            raise ValueError(
                f"{self.ablation_variant!r} is not a unified MCMoE++ variant; "
                "this package exposes only unified MCMoE++ variants"
            )
        text_dim = getattr(config, "text_dim", None) or in_dim
        visual_dim = getattr(config, "visual_dim", None) or 768
        audio_dim = getattr(config, "audio_dim", None) or 1024
        self.in_proj = nn.ModuleDict(
            {
                "v": self._build_proj(text_dim, hidden_dim),
                "a": self._build_proj(visual_dim, hidden_dim),
                "f": self._build_proj(audio_dim, hidden_dim),
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
                "v": AGMG(hidden_dim),
                "a": AGMG(hidden_dim),
                "f": AGMG(hidden_dim),
            }
        )
        self.router_mix = Mlp(3 * hidden_dim, hidden_dim // 2, 2)
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
        self.register_buffer("weight", torch.linspace(0, 1, n_query))
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Variant-only parameters are instantiated only when they are part of
        # the real forward path, so parameter counts and optimization are honest.
        if self.ablation_variant == "wo_ma_router":
            # One global residual-coefficient pair shared by every stream and
            # sample: no feature, stream identity, or availability input.
            self.global_residual_logits = nn.Parameter(torch.zeros(2))
            del self.router_mix
        elif self.ablation_variant == "wo_anchor_residual":
            # Keep the exact MA conditioning input, but predict an ordinary
            # three-expert softmax instead of two anchor residual coefficients.
            self.router_mix = Mlp(3 * hidden_dim, hidden_dim // 2, 3)

        if self.ablation_variant == "ma_moe":
            del self.experts["s"]
            del self.stream_reliability
            del self.shared_specific_gate
            del self.gen_cap_prior
            del self.shared_cap_prior
        elif self.ablation_variant == "wo_shared_expert":
            del self.experts["s"]
        elif self.ablation_variant == "wo_coord":
            del self.stream_reliability
            del self.shared_specific_gate
            del self.gen_cap_prior
            del self.shared_cap_prior
        elif self.ablation_variant == "wo_gamma_stream":
            del self.stream_reliability
            del self.gen_cap_prior
        elif self.ablation_variant == "wo_beta_global":
            del self.shared_specific_gate
            del self.shared_cap_prior

    def _score_from_query_features(self, fea):
        s = self.regressor(fea)
        s = torch.diagonal(s, dim1=-2, dim2=-1)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        return torch.sum(self.weight.unsqueeze(0).repeat(fea.size(0), 1) * norm_s, dim=1)

    def _conversation_score_head(self, fusion):
        b, t, _ = fusion.shape
        prototype = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        prototype = self.pos_encoder(prototype)
        outs = []
        for step in range(t):
            fea_step, _ = self.transformer.decoder(prototype, fusion[:, step : step + 1, :])
            outs.append(self._score_from_query_features(fea_step).unsqueeze(1))
        return torch.cat(outs, dim=1)

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

    def _expand_mask_tensor(self, mask_tensor, b, t=None):
        if mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).repeat(b, 1)
        elif mask_tensor.size(0) == 1:
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.repeat(b, 1)
            elif mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.repeat(b, 1, 1)
        if mask_tensor.dim() == 3 and t is not None and mask_tensor.size(1) == 1:
            mask_tensor = mask_tensor.repeat(1, t, 1)
        return mask_tensor

    def _normalize_mask_tensor(self, mask, b, device, t=None):
        if torch.is_tensor(mask):
            mask_tensor = mask.to(device=device, dtype=torch.float32)
        else:
            mask_tensor = torch.tensor(mask, device=device, dtype=torch.float32)
        return self._expand_mask_tensor(mask_tensor, b, t=t).clamp(0.0, 1.0)

    def _build_router_input(self, feat, mask_tensor, stream_idx):
        b, t, _ = feat.shape
        mask_tensor = self._expand_mask_tensor(mask_tensor, b, t=t)
        if mask_tensor.dim() == 3:
            mask_emb = self.mask_embed(mask_tensor.float())
        else:
            mask_emb = self.mask_embed(mask_tensor.float()).unsqueeze(1).repeat(1, t, 1)
        stream_ids = torch.full((b, t), stream_idx, dtype=torch.long, device=feat.device)
        stream_emb = self.stream_embed(stream_ids)
        return torch.cat([feat, stream_emb, mask_emb], dim=-1)

    def _conditioned_router(self, feat, mask_tensor, stream_idx):
        return torch.sigmoid(
            self.router_mix(self._build_router_input(feat, mask_tensor, stream_idx))
        )

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

    def _build_shared_feat(self, v_mask, a_mask, f_mask):
        shared_base = (v_mask + a_mask + f_mask) / 3.0
        shared_feat = self.experts["s"](shared_base.transpose(1, 2)).transpose(1, 2)
        return shared_feat

    def _get_mask_id(self, mask_tensor):
        """Map a visibility mask to mask_id in [0, 7].

        Supports (B, 3) and (B, T, 3) masks.
        mask_id = v * 4 + a * 2 + f.
        """
        if mask_tensor.dim() == 3:
            return (
                mask_tensor[:, :, 0].long() * 4
                + mask_tensor[:, :, 1].long() * 2
                + mask_tensor[:, :, 2].long()
            ).clamp(0, 7)
        return (
            mask_tensor[:, 0].long() * 4 + mask_tensor[:, 1].long() * 2 + mask_tensor[:, 2].long()
        ).clamp(0, 7)

    def _stream_reliability_gate(self, feat, mask_tensor, stream_idx):
        b, t, c = feat.shape
        mask_tensor = self._expand_mask_tensor(mask_tensor, b, t=t)
        gate_input = self._build_router_input(feat, mask_tensor, stream_idx)
        raw_rel = torch.sigmoid(self.stream_reliability(gate_input))
        if mask_tensor.dim() == 3:
            real_flag = mask_tensor[:, :, stream_idx : stream_idx + 1]
        else:
            real_flag = mask_tensor[:, stream_idx : stream_idx + 1].view(b, 1, 1)
        # Learnable mask-conditioned cap. Initialized to n_vis / 3, then learnable.
        mask_id = self._get_mask_id(mask_tensor)
        gen_cap = torch.sigmoid(self.gen_cap_prior(mask_id))
        if gen_cap.dim() == 2:
            gen_cap = gen_cap.view(b, 1, 1)
        reliability = real_flag * raw_rel + (1.0 - real_flag) * raw_rel * gen_cap
        return reliability

    def _global_shared_specific_fusion(self, specific_global, shared_feat, mask_tensor):
        b, t, c = specific_global.shape
        mask_tensor = self._expand_mask_tensor(mask_tensor, b, t=t)
        if mask_tensor.dim() == 3:
            mask_emb = self.mask_embed(mask_tensor.float())
        else:
            mask_emb = self.mask_embed(mask_tensor.float()).unsqueeze(1).repeat(1, t, 1)
        gate_input = torch.cat([specific_global, shared_feat, mask_emb], dim=-1)
        raw_shared_ratio = torch.sigmoid(self.shared_specific_gate(gate_input))
        # Learnable mask-conditioned cap. Initialized to n_vis / 3, then learnable.
        mask_id = self._get_mask_id(mask_tensor)
        shared_cap = torch.sigmoid(self.shared_cap_prior(mask_id))
        if shared_cap.dim() == 2:
            shared_cap = shared_cap.view(b, 1, 1)
        shared_ratio = raw_shared_ratio * shared_cap
        fusion = specific_global + shared_ratio * (shared_feat - specific_global)
        return fusion, shared_ratio

    def computation_path(self):
        """Return the declarative path used for checkpoint/config auditing."""
        variant = self.ablation_variant
        return {
            "variant": variant,
            "router": (
                "global_residual_coefficients"
                if variant == "wo_ma_router"
                else (
                    "ma_conditioned_three_way_softmax"
                    if variant == "wo_anchor_residual"
                    else "ma_conditioned_residual_coefficients"
                )
            ),
            "expert_mixing": (
                "ordinary_softmax_sum" if variant == "wo_anchor_residual" else "anchor_residual"
            ),
            "stream_gate": (
                "fixed_1" if variant in {"ma_moe", "wo_coord", "wo_gamma_stream"} else "acdg_gamma"
            ),
            "shared_representation": (
                "none"
                if variant == "ma_moe"
                else (
                    "completed_stream_mean"
                    if variant == "wo_shared_expert"
                    else "learned_shared_expert"
                )
            ),
            "global_gate": (
                "none"
                if variant == "ma_moe"
                else "fixed_0.5"
                if variant in {"wo_coord", "wo_beta_global"}
                else "acdg_beta"
            ),
            "cfm": True,
        }

    def _expert_outputs(self, v_mask, a_mask, f_mask):
        outputs = []
        for stream in (v_mask, a_mask, f_mask):
            outputs.append(
                tuple(
                    self.experts[key](stream.transpose(1, 2)).transpose(1, 2)
                    for key in ("v", "a", "f")
                )
            )
        return outputs

    @staticmethod
    def _ordinary_expert_sum(experts, weights):
        stacked = torch.stack(experts, dim=2)
        return (stacked * weights.unsqueeze(-1)).sum(dim=2)

    def _specific_and_global_path(self, v_mask, a_mask, f_mask, mask_tensor):
        """Run the sole variant-controlled MA-MoE/MASEC path.

        AGMG completion, the downstream CFM block, decoder, losses, dataset
        protocol, and checkpoint selection are outside this function and are
        therefore identical for every unified variant.
        """
        streams = (v_mask, a_mask, f_mask)
        expert_outputs = self._expert_outputs(v_mask, a_mask, f_mask)
        specific_pre = []

        for stream_idx, (stream, experts) in enumerate(zip(streams, expert_outputs)):
            if self.ablation_variant == "wo_ma_router":
                coeff = torch.sigmoid(self.global_residual_logits)
                weights = coeff.view(1, 1, 2).expand(stream.size(0), stream.size(1), 2)
            elif self.ablation_variant == "wo_anchor_residual":
                weights = F.softmax(
                    self.router_mix(self._build_router_input(stream, mask_tensor, stream_idx)),
                    dim=-1,
                )
            else:
                weights = self._conditioned_router(stream, mask_tensor, stream_idx)
            if self.ablation_variant == "wo_anchor_residual":
                mixed = self._ordinary_expert_sum(experts, weights)
            else:
                mixed = self._complementary_specific_fusion(
                    *experts, weights, anchor_idx=stream_idx
                )
            specific_pre.append(mixed)

        if self.ablation_variant in {"ma_moe", "wo_coord", "wo_gamma_stream"}:
            reliability = [torch.ones_like(feat[..., :1]) for feat in specific_pre]
        else:
            reliability = [
                self._stream_reliability_gate(feat, mask_tensor, stream_idx)
                for stream_idx, feat in enumerate(specific_pre)
            ]
        specific = [gate * feat for gate, feat in zip(reliability, specific_pre)]

        specific_cat = torch.cat(specific, dim=-1)
        specific_global = self.fusion(specific_cat.transpose(1, 2)).transpose(1, 2)

        if self.ablation_variant == "ma_moe":
            fusion = specific_global
        else:
            if self.ablation_variant == "wo_shared_expert":
                shared_feat = (v_mask + a_mask + f_mask) / 3.0
            else:
                shared_feat = self._build_shared_feat(v_mask, a_mask, f_mask)

            if self.ablation_variant in {"wo_coord", "wo_beta_global"}:
                shared_ratio = torch.full_like(specific_global[..., :1], 0.5)
                fusion = specific_global + shared_ratio * (shared_feat - specific_global)
            else:
                fusion, shared_ratio = self._global_shared_specific_fusion(
                    specific_global, shared_feat, mask_tensor
                )

        return fusion, specific_pre

    def _encode_with_mask(self, video, audio, flow, mask_tensor, use_recon_loss):
        proj_v = self.transformer.encoder(self.in_proj["v"](video.transpose(1, 2)).transpose(1, 2))
        proj_a = self.transformer.encoder(self.in_proj["a"](audio.transpose(1, 2)).transpose(1, 2))
        proj_f = self.transformer.encoder(self.in_proj["f"](flow.transpose(1, 2)).transpose(1, 2))

        b, t, _ = video.shape
        mask_tensor = self._expand_mask_tensor(mask_tensor, b, t=t)
        if mask_tensor.dim() == 3:
            v_flag = mask_tensor[:, :, 0:1]
            a_flag = mask_tensor[:, :, 1:2]
            f_flag = mask_tensor[:, :, 2:3]
        else:
            v_flag = mask_tensor[:, 0].view(-1, 1, 1)
            a_flag = mask_tensor[:, 1].view(-1, 1, 1)
            f_flag = mask_tensor[:, 2].view(-1, 1, 1)
        v_mask = proj_v * v_flag
        a_mask = proj_a * a_flag
        f_mask = proj_f * f_flag
        recon_loss = proj_v.new_tensor(0.0)

        def available_context(candidates):
            context = [feat for flag, feat in candidates if bool((flag > 0.5).any().item())]
            return context if context else [torch.zeros_like(v_mask)]

        if (v_flag < 0.5).any():
            v_gen = self.generators["v"](
                available_context([(a_flag, a_mask), (f_flag, f_mask)]), torch.zeros_like(v_mask)
            )
            if use_recon_loss:
                recon_loss = recon_loss + F.mse_loss(
                    v_gen[v_flag.expand_as(v_gen) < 0.5], proj_v[v_flag.expand_as(proj_v) < 0.5]
                )
            v_mask = v_flag * v_mask + (1.0 - v_flag) * v_gen
        if (a_flag < 0.5).any():
            a_gen = self.generators["a"](
                available_context([(v_flag, v_mask), (f_flag, f_mask)]), torch.zeros_like(a_mask)
            )
            if use_recon_loss:
                recon_loss = recon_loss + F.mse_loss(
                    a_gen[a_flag.expand_as(a_gen) < 0.5], proj_a[a_flag.expand_as(proj_a) < 0.5]
                )
            a_mask = a_flag * a_mask + (1.0 - a_flag) * a_gen
        if (f_flag < 0.5).any():
            f_gen = self.generators["f"](
                available_context([(v_flag, v_mask), (a_flag, a_mask)]), torch.zeros_like(f_mask)
            )
            if use_recon_loss:
                recon_loss = recon_loss + F.mse_loss(
                    f_gen[f_flag.expand_as(f_gen) < 0.5], proj_f[f_flag.expand_as(proj_f) < 0.5]
                )
            f_mask = f_flag * f_mask + (1.0 - f_flag) * f_gen

        return proj_v, proj_a, proj_f, v_mask, a_mask, f_mask, recon_loss

    def forward(self, video, audio, flow, mask):
        if self.training:
            return self.forward_train(video, audio, flow, mask)
        else:
            return self.forward_test(video, audio, flow, mask)

    def forward_train(self, video, audio, flow, mask):
        b, t, _ = video.shape
        mask_tensor = self._normalize_mask_tensor(mask, b, video.device, t=t)
        proj_v, proj_a, proj_f, v_mask, a_mask, f_mask, recon_loss = self._encode_with_mask(
            video, audio, flow, mask_tensor, use_recon_loss=True
        )
        fusion, (specific_v_pre, specific_a_pre, specific_f_pre) = self._specific_and_global_path(
            v_mask, a_mask, f_mask, mask_tensor
        )
        prototype = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        prototype = self.pos_encoder(prototype)
        fea, _ = self.transformer.decoder(prototype, fusion)
        out = self._conversation_score_head(fusion)
        expert_v = self.experts["v"](proj_v.transpose(1, 2)).transpose(1, 2)
        expert_a = self.experts["a"](proj_a.transpose(1, 2)).transpose(1, 2)
        expert_f = self.experts["f"](proj_f.transpose(1, 2)).transpose(1, 2)
        recon_loss += F.kl_div(
            torch.log_softmax(specific_v_pre, -1),
            torch.softmax(expert_v, -1),
            reduction="mean",
        )
        recon_loss += F.kl_div(
            torch.log_softmax(specific_a_pre, -1),
            torch.softmax(expert_a, -1),
            reduction="mean",
        )
        recon_loss += F.kl_div(
            torch.log_softmax(specific_f_pre, -1),
            torch.softmax(expert_f, -1),
            reduction="mean",
        )
        return {
            "output": out,
            "embed": fea,
            "embed2": None,
            "recon_loss": recon_loss,
        }

    def forward_test(self, video, audio, flow, mask):
        b, t, _ = video.shape
        mask_tensor = self._normalize_mask_tensor(mask, b, video.device, t=t)
        _, _, _, v_mask, a_mask, f_mask, _ = self._encode_with_mask(
            video, audio, flow, mask_tensor, use_recon_loss=False
        )
        fusion, _ = self._specific_and_global_path(v_mask, a_mask, f_mask, mask_tensor)
        return {"output": self._conversation_score_head(fusion)}
