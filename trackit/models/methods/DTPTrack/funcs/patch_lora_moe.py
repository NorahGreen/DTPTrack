from typing import Sequence, List, Optional

import torch
import torch.nn as nn
from ..modules.lora_moe import LinearWithLoRAMoE, LinearWithLoRAMoE_QKVFused, LinearWithLoRAMoE_KVFused
from .vit_lora_utils import find_all_frozen_nn_linear_names



def apply_lora_moe(model: nn.Module, lora_module_names: List[str],
                   r: int, num_experts: int,
                   top_k: int,
                   use_shared_expert: bool, load_balance_coeff: Optional[float]):
    for lora_module_name in lora_module_names:
        *parent_path, child_module_name = lora_module_name.split(".")
        parent_module = model.get_submodule(".".join(parent_path))

        if child_module_name == 'qkv':
            wrapper_cls = LinearWithLoRAMoE_QKVFused
        elif child_module_name == 'kv':
            wrapper_cls = LinearWithLoRAMoE_KVFused
        else:
            wrapper_cls = LinearWithLoRAMoE
        setattr(parent_module, child_module_name,
                wrapper_cls(getattr(parent_module, child_module_name), r, num_experts,
                            top_k, use_shared_expert, load_balance_coeff))


def enable_lora_moe_(self: nn.Module, r: int, num_experts: int,
                     top_k: int,
                     use_shared_expert: bool, load_balance_coeff: Optional[float]):
    for i_layer, block in enumerate(self.blocks):
        linear_names = find_all_frozen_nn_linear_names(block)
        apply_lora_moe(block, linear_names, r, num_experts,
                       top_k, use_shared_expert, load_balance_coeff)
