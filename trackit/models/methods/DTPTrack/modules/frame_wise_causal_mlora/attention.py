from typing import Tuple, Sequence, Optional, Mapping, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Attention

from trackit.models.methods.DTPTrack.modules.rope_2d import apply_rope
from ..mlora.mlora import mLoRALinear


class QKVLinearWithMLora(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super().__init__()
        self.q = mLoRALinear(in_features, out_features, bias=bias, **kwargs)
        self.k = mLoRALinear(in_features, out_features, bias=bias, **kwargs)
        self.v = mLoRALinear(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x: torch.Tensor, task_idx: int) -> torch.Tensor:
        return torch.cat([self.q(x, task_idx), self.k(x, task_idx), self.v(x, task_idx)], dim=-1)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], **kwargs: Any
    ):
        state_dict = dict(state_dict)
        dim = self.q.in_features
        state_dict['q.weight'] = state_dict['weight'][:dim]
        state_dict['k.weight'] = state_dict['weight'][dim:2 * dim]
        state_dict['v.weight'] = state_dict['weight'][2 * dim:]
        del state_dict['weight']
        if 'bias' in state_dict:
            state_dict['q.bias'] = state_dict['bias'][:dim]
            state_dict['k.bias'] = state_dict['bias'][dim:2 * dim]
            state_dict['v.bias'] = state_dict['bias'][2 * dim:]
            del state_dict['bias']
        return super().load_state_dict(state_dict, **kwargs)


class ChunkedCausalAttention_MoE(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused_attn: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = QKVLinearWithMLora(dim, dim, bias=qkv_bias, **kwargs)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = mLoRALinear(dim, dim, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, expert_idx: Sequence[int] | int, chunk_sizes: Sequence[int],
                freqs_cis: Sequence[torch.Tensor] | None = None,
                past_kv_enable_bits: Optional[Sequence[bool]] = None) -> torch.Tensor:
        x_chunks = x.split(chunk_sizes, dim=1)
        k_list = []
        v_list = []

        x_out = []
        for i in range(len(chunk_sizes)):
            x = x_chunks[i]
            B, N, C = x.shape
            this_i_expert = expert_idx if isinstance(expert_idx, int) else expert_idx[i]
            qkv = self.qkv(x, this_i_expert).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            if freqs_cis is not None:
                q, k = apply_rope(q, freqs_cis[i], -3), apply_rope(k, freqs_cis[i], -3)
            k_list.append(k)
            v_list.append(v)
            if past_kv_enable_bits is None or past_kv_enable_bits[i]:
                k = torch.cat(k_list, dim=-2)
                v = torch.cat(v_list, dim=-2)

            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q_chunk = q * self.scale
                attn = q_chunk @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x, this_i_expert)
            x = self.proj_drop(x)
            x_out.append(x)
        return torch.cat(x_out, dim=1)

    @staticmethod
    def copy_from_std_attn(other: Attention, **kwargs: Any) -> 'ChunkedCausalAttention_MoE':
        attn = ChunkedCausalAttention_MoE.__new__(ChunkedCausalAttention_MoE)
        nn.Module.__init__(attn)
        attn.num_heads = other.num_heads
        attn.head_dim = other.head_dim
        dim = other.qkv.in_features
        attn.qkv = QKVLinearWithMLora(dim, dim, bias=other.qkv.bias is not None, **kwargs)
        attn.qkv.load_state_dict(other.qkv.state_dict(), strict=False)
        attn.q_norm = other.q_norm
        attn.k_norm = other.k_norm
        attn.proj = mLoRALinear(dim, dim, **kwargs)
        attn.proj.load_state_dict(other.proj.state_dict(), strict=False)
        attn.fused_attn = True
        attn.attn_drop = other.attn_drop
        attn.proj_drop = other.proj_drop
        attn.scale = other.scale
        return attn
