# from https://github.com/facebookresearch/dinov2/blob/6a6261546c3357f2c243a60cfafa6607f84efcb7/dinov2/models/vision_transformer.py#L164
import torch
import torch.nn as nn


def interpolate_pos_encoding(pos_embed, old_w, old_h, new_w, new_h):
    dim = pos_embed.shape[-1]
    w0, h0 = new_w + 0.1, new_h + 0.1

    old_dtype = pos_embed.dtype
    pos_embed = nn.functional.interpolate(
        pos_embed.to(torch.float32).reshape(1, old_h, old_w, dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / old_w, h0 / old_h),
        mode="bicubic",
    )

    assert int(w0) == pos_embed.shape[-2] and int(h0) == pos_embed.shape[-1]
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim).to(old_dtype)
    return pos_embed
