import torch
from torch import nn, Tensor
from typing import Callable
from ...attention.with_rotary_emb import Attention
from ...mlp import Mlp
from ...layer_scale import LayerScale
from ...drop_path import DropPath


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), freqs_cis)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    @classmethod
    def build_from_standard_block(cls, standard_block, with_attn: bool = True):
        block = cls.__new__(cls)
        nn.Module.__init__(block)
        block.norm1 = standard_block.norm1
        block.attn = standard_block.attn
        block.ls1 = standard_block.ls1
        block.drop_path1 = standard_block.drop_path1
        block.norm2 = standard_block.norm2
        block.mlp = standard_block.mlp
        block.ls2 = standard_block.ls2
        block.drop_path2 = standard_block.drop_path2

        if with_attn:
            from ...attention.with_rotary_emb import Attention
            block.attn = Attention.build_from_standard_attention(block.attn)

        return block
