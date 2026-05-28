from typing import Sequence

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.layers.mlp import Mlp
from ..mlora.mlora import mLoRALinear


class Mlp_MoE(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            **kwargs
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = mLoRALinear(in_features, hidden_features, bias=bias[0], **kwargs)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 =  mLoRALinear(hidden_features, out_features, bias=bias[1], **kwargs)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor, expert_idx: Sequence[int] | int, chunk_sizes: Sequence[int] | None = None):
        if chunk_sizes is not None:
            x_chunks = x.split(chunk_sizes, dim=1)
            x_out = []
            for i, x_chunk in enumerate(x_chunks):
                this_moe_idx = expert_idx if isinstance(expert_idx, int) else expert_idx[i]
                x_chunk = self.fc1(x_chunk, this_moe_idx)
                x_chunk = self.act(x_chunk)
                x_chunk = self.drop1(x_chunk)
                x_chunk = self.norm(x_chunk)
                x_chunk = self.fc2(x_chunk, this_moe_idx)
                x_chunk = self.drop2(x_chunk)
                x_out.append(x_chunk)
            return torch.cat(x_out, dim=1)
        else:
            # assert isinstance(expert_idx, int)
            x = self.fc1(x, expert_idx)
            x = self.act(x)
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x, expert_idx)
            x = self.drop2(x)
            return x

    @staticmethod
    def copy_from_std_block(other: Mlp, **kwargs):
        mlp = Mlp_MoE.__new__(Mlp_MoE)
        nn.Module.__init__(mlp)
        mlp.act = other.act
        mlp.fc1 = mLoRALinear(other.fc1.in_features, other.fc1.out_features, bias=other.fc1.bias is not None, **kwargs)
        mlp.fc1.load_state_dict(other.fc1.state_dict(), strict=False)
        mlp.fc2 = mLoRALinear(other.fc2.in_features, other.fc2.out_features, bias=other.fc2.bias is not None, **kwargs)
        mlp.fc2.load_state_dict(other.fc2.state_dict(), strict=False)
        mlp.norm = other.norm
        mlp.drop1 = other.drop1
        mlp.drop2 = other.drop2
        return mlp
