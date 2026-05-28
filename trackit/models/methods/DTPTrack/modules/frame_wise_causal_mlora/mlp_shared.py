from typing import Sequence

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.layers.mlp import Mlp
from ..lora import LinearWithLoRA


class Mlp_MoE_Shared(nn.Module):
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

        self.fc1 = LinearWithLoRA(nn.Linear(in_features, hidden_features, bias=bias[0]), r=kwargs['r'], alpha=kwargs['r'], dropout=0)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = LinearWithLoRA(nn.Linear(hidden_features, out_features, bias=bias[1]), r=kwargs['r'], alpha=kwargs['r'], dropout=0)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor, expert_idx: Sequence[int] | int, chunk_sizes: Sequence[int] | None = None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    @staticmethod
    def copy_from_std_block(other: Mlp, **kwargs):
        mlp = Mlp_MoE_Shared.__new__(Mlp_MoE_Shared)
        nn.Module.__init__(mlp)
        mlp.act = other.act
        mlp.fc1 = LinearWithLoRA(other.fc1, r=kwargs['r'], alpha=kwargs['r'], dropout=0)
        mlp.fc2 = LinearWithLoRA(other.fc2, r=kwargs['r'], alpha=kwargs['r'], dropout=0)
        mlp.norm = other.norm
        mlp.drop1 = other.drop1
        mlp.drop2 = other.drop2
        return mlp
