from typing import Optional

import torch
import math
import torch.nn as nn
from functools import partial
from torch.nn.init import trunc_normal_
from .moe import MoE


class LoRAExpertLayer(nn.Module):
    def __init__(self, num_experts: int, in_dim: int, out_dim: int, r: int,
                 init_method: str = 'bert'):
        super().__init__()
        self.A = nn.Parameter(torch.empty(num_experts, in_dim, r))
        self.B = nn.Parameter(torch.empty(num_experts, r, out_dim))
        self.r = r
        if init_method == 'lora':
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        elif init_method == 'gaussian':
            nn.init.normal_(self.A, std=1. / self.r)
            nn.init.zeros_(self.B)
        elif init_method == 'bert':
            trunc_normal_(self.A, std=.02)
            trunc_normal_(self.B, std=.02)
        else:
            raise ValueError(f'Unknown init method: {init_method}')

    def forward(self,
                x: torch.Tensor,
                num_local_tokens_per_expert: torch.Tensor | list[int] | None = None):
        if num_local_tokens_per_expert is not None:
            # a tuple of tensors indexed by experts
            # each with shape (tokens_per_expert(varying), dim)
            x = torch.split(
                x,
                split_size_or_sections=num_local_tokens_per_expert,
                dim=0,
            )
            out_experts_splits = []
            for expert_idx, x_expert in enumerate(x):
                A, B = (
                    self.A[expert_idx],
                    self.B[expert_idx],
                )
                h = torch.matmul(x_expert, A)
                h = torch.matmul(h, B)
                out_experts_splits.append(h)
            out = torch.cat(out_experts_splits, dim=0)
        else:
            # x shape (num_experts, tokens_per_expert, dim)
            h = torch.bmm(x, self.A)
            out = torch.bmm(h, self.B)

        return out


class LinearWithLoRAMoE(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, num_experts: int,
                 top_k: int,
                 use_shared_expert: bool, load_balance_coeff: Optional[float],
                 init_method: str='bert'):
        super().__init__()
        self.linear = linear
        self.lora_moe = MoE(partial(LoRAExpertLayer,
                                    in_dim=linear.in_features, out_dim=linear.out_features,
                                    r=r, init_method=init_method),
                            dim=linear.in_features, num_experts=num_experts,
                            top_k=top_k,
                            use_shared_expert=use_shared_expert,
                            load_balance_coeff=load_balance_coeff)

    def forward(self, x: torch.Tensor):
        return self.linear(x) + self.lora_moe(x)


class LinearWithLoRAMoE_QKVFused(nn.Module):
    def __init__(self, qkv_linear: nn.Linear, r: int, num_experts: int,
                 top_k: int,
                 use_shared_expert: bool, load_balance_coeff: Optional[float],
                 init_method: str='bert',
                 target_q: bool = True, target_k: bool = True, target_v: bool = True):
        super().__init__()
        dim = qkv_linear.in_features
        bias = qkv_linear.bias is not None
        q = nn.Linear(dim, dim, bias, device=qkv_linear.weight.device, dtype=qkv_linear.weight.dtype)
        k = nn.Linear(dim, dim, bias, device=qkv_linear.weight.device, dtype=qkv_linear.weight.dtype)
        v = nn.Linear(dim, dim, bias, device=qkv_linear.weight.device, dtype=qkv_linear.weight.dtype)
        q.weight.data.copy_(qkv_linear.weight.data[:dim])
        k.weight.data.copy_(qkv_linear.weight.data[dim:2 * dim])
        v.weight.data.copy_(qkv_linear.weight.data[2 * dim:])
        q.weight.requires_grad = k.weight.requires_grad = v.weight.requires_grad = qkv_linear.weight.requires_grad
        if bias:
            q.bias.data.copy_(qkv_linear.bias.data[:dim])
            k.bias.data.copy_(qkv_linear.bias.data[dim:2 * dim])
            v.bias.data.copy_(qkv_linear.bias.data[2 * dim:])
            q.bias.requires_grad = k.bias.requires_grad = v.bias.requires_grad = qkv_linear.bias.requires_grad

        if target_q:
            self.q = LinearWithLoRAMoE(q, r, num_experts, top_k, use_shared_expert, load_balance_coeff, init_method)
        else:
            self.q = q
        if target_k:
            self.k = LinearWithLoRAMoE(k, r, num_experts, top_k, use_shared_expert, load_balance_coeff, init_method)
        else:
            self.k = k
        if target_v:
            self.v = LinearWithLoRAMoE(v, r, num_experts, top_k, use_shared_expert, load_balance_coeff, init_method)
        else:
            self.v = v

    def forward(self, x: torch.Tensor):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat((q, k, v), dim=-1)


class LinearWithLoRAMoE_KVFused(nn.Module):
    def __init__(self, kv_linear: nn.Linear, r: int, num_experts: int,
                 top_k: int,
                 use_shared_expert: bool, load_balance_coeff: Optional[float],
                 init_method: str='bert',
                 target_k: bool = True, target_v: bool = True):
        super().__init__()
        dim = kv_linear.in_features
        bias = kv_linear.bias is not None
        k = nn.Linear(dim, dim, bias, device=kv_linear.weight.device, dtype=kv_linear.weight.dtype)
        v = nn.Linear(dim, dim, bias, device=kv_linear.weight.device, dtype=kv_linear.weight.dtype)
        k.weight.data.copy_(kv_linear.weight.data[:dim])
        v.weight.data.copy_(kv_linear.weight.data[dim:2 * dim])
        k.weight.requires_grad = v.weight.requires_grad = kv_linear.weight.requires_grad
        if bias:
            k.bias.data.copy_(kv_linear.bias.data[:dim])
            v.bias.data.copy_(kv_linear.bias.data[dim:2 * dim])
            k.bias.requires_grad = v.bias.requires_grad = kv_linear.bias.requires_grad

        if target_k:
            self.k = LinearWithLoRAMoE(k, r, num_experts, top_k, use_shared_expert, load_balance_coeff, init_method)
        else:
            self.k = k
        if target_v:
            self.v = LinearWithLoRAMoE(v, r, num_experts, top_k, use_shared_expert, load_balance_coeff, init_method)
        else:
            self.v = v

    def forward(self, x: torch.Tensor):
        k = self.k(x)
        v = self.v(x)
        return torch.cat((k, v), dim=-1)
