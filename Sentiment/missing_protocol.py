import torch


def sample_utterance_mask(batch_size, seq_len, missing_rate, device, generator=None):
    """Sample GCNet-style per-utterance modality masks.

    Returns a float tensor of shape [B, T, 3]. Each utterance keeps at least one
    modality, as in GCNet's random_mask protocol.
    """
    missing_rate = float(missing_rate)
    input_len = int(batch_size) * int(seq_len)
    if input_len <= 0:
        return torch.ones(batch_size, seq_len, 3, device=device)

    view_num = 3
    one_rate = 1.0 - missing_rate
    if one_rate <= (1.0 / view_num):
        view = torch.randint(view_num, (input_len,), device=device, generator=generator)
        matrix = torch.nn.functional.one_hot(view, num_classes=view_num).float()
        return matrix.view(batch_size, seq_len, view_num)

    if one_rate >= 1.0:
        return torch.ones(batch_size, seq_len, view_num, device=device)

    all_len = max(input_len, 32)
    view = torch.randint(view_num, (all_len,), device=device, generator=generator)
    matrix = torch.nn.functional.one_hot(view, num_classes=view_num).bool()
    target_ones = int(round(view_num * all_len * one_rate))
    extra_ones = target_ones - all_len
    if extra_ones > 0:
        zero_pos = (matrix == 0).nonzero(as_tuple=False)
        extra_ones = min(extra_ones, zero_pos.size(0))
        chosen = torch.randperm(zero_pos.size(0), device=device, generator=generator)[:extra_ones]
        selected = zero_pos[chosen]
        matrix[selected[:, 0], selected[:, 1]] = True

    matrix = matrix[:input_len].float()
    return matrix.view(batch_size, seq_len, view_num)


def apply_modality_mask(text_feat, visual_feat, audio_feat, mask):
    """Zero modalities using mask columns [T, V, A].

    Supports either [B, 3] sample-level masks or [B, T, 3] utterance-level masks.
    """
    mask = mask.to(device=text_feat.device, dtype=text_feat.dtype)
    if mask.dim() == 2:
        text_gate = mask[:, 0].view(-1, 1, 1)
        visual_gate = mask[:, 1].view(-1, 1, 1)
        audio_gate = mask[:, 2].view(-1, 1, 1)
    elif mask.dim() == 3:
        text_gate = mask[:, :, 0:1]
        visual_gate = mask[:, :, 1:2]
        audio_gate = mask[:, :, 2:3]
    else:
        raise ValueError(f"Unexpected modality mask shape: {tuple(mask.shape)}")
    text_feat = text_feat * text_gate
    visual_feat = visual_feat * visual_gate
    audio_feat = audio_feat * audio_gate
    return text_feat, visual_feat, audio_feat


def actual_missing_rate(mask):
    return float(1.0 - mask.sum().item() / float(mask.numel()))
