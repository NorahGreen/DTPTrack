from torch import Tensor
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from flash_attn.layers.rotary import RotaryEmbedding


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rotary_emb = RotaryEmbedding(dim // num_heads)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = self.rotary_emb(qkv)
        x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop.p if self.attn_drop.training else 0)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    @classmethod
    def build_from_standattn(cls, other):
        attn = cls.__new__(cls)
        nn.Module.__init__(attn)
        attn.num_heads = other.num_heads
        attn.scale = other.scale
        attn.qkv = other.qkv
        attn.attn_drop = other.attn_drop
        attn.proj = other.proj
        attn.proj_drop = other.proj_drop
        attn.rotary_emb = RotaryEmbedding(other.qkv.in_features // other.num_heads)
        return attn
