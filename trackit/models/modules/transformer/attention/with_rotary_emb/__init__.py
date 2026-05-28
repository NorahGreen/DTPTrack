import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from ...rotary_emb import apply_rotary_emb


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            fused_attn: bool = True
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = fused_attn

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @classmethod
    def build_from_standard_attention(cls, standard_attn):
        attention = cls.__new__(cls)
        nn.Module.__init__(attention)
        attention.num_heads = standard_attn.num_heads
        attention.scale = standard_attn.scale
        attention.qkv = standard_attn.qkv
        attention.attn_drop = standard_attn.attn_drop
        attention.proj = standard_attn.proj
        attention.proj_drop = standard_attn.proj_drop
        if hasattr(standard_attn, 'fused_attn'):
            attention.fused_attn = standard_attn.fused_attn
        else:
            attention.fused_attn = True
        return attention
