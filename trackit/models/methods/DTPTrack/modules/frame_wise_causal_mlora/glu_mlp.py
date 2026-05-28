from functools import partial
from typing import Sequence

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.layers.mlp import GluMlp
from ..mlora.mlora import mLoRALinear


class GluMlp_MoE(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Sigmoid,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1.bias is not None:
            nn.init.ones_(self.fc1.bias[self.fc1.bias.shape[0] // 2:])
        nn.init.normal_(self.fc1.weight[self.fc1.weight.shape[0] // 2:], std=1e-6)

    def forward(self, x, expert_idx: Sequence[int] | int, chunk_sizes: Sequence[int] | None = None):
        if chunk_sizes is not None:
            x_chunks = x.split(chunk_sizes, dim=1)
            x_out = []
            for i, x_chunk in enumerate(x_chunks):
                this_moe_idx = expert_idx if isinstance(expert_idx, int) else expert_idx[i]
                x_chunk = self.fc1(x_chunk, this_moe_idx)
                x1, x2 = x_chunk.chunk(2, dim=self.chunk_dim)
                x_chunk = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
                x_chunk = self.drop1(x_chunk)
                x_chunk = self.norm(x_chunk)
                x_chunk = self.fc2(x_chunk, this_moe_idx)
                x_chunk = self.drop2(x_chunk)
                x_out.append(x_chunk)
            return torch.cat(x_out, dim=1)
        else:
            x = self.fc1(x)
            x1, x2 = x.chunk(2, dim=self.chunk_dim)
            x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x

    @staticmethod
    def build_from_std_block(other: GluMlp, **kwargs):
        glu_mlp = GluMlp_MoE.__new__(GluMlp_MoE)
        nn.Module.__init__(glu_mlp)
        assert isinstance(other, GluMlp)
        assert isinstance(other.fc1, nn.Linear) and isinstance(other.fc2, nn.Linear)
        glu_mlp.fc1 = mLoRALinear(other.fc1.in_features, other.fc1.out_features, bias=other.fc1.bias is not None, **kwargs)
        glu_mlp.fc1.load_state_dict(other.fc1.state_dict(), strict=False)
        glu_mlp.fc2 = mLoRALinear(other.fc2.in_features, other.fc2.out_features, bias=other.fc2.bias is not None, **kwargs)
        glu_mlp.fc2.load_state_dict(other.fc2.state_dict(), strict=False)
        glu_mlp.act = other.act
        glu_mlp.chunk_dim = other.chunk_dim
        glu_mlp.gate_last = other.gate_last
        glu_mlp.drop1 = other.drop1
        glu_mlp.norm = other.norm
        glu_mlp.drop2 = other.drop2
        return glu_mlp
