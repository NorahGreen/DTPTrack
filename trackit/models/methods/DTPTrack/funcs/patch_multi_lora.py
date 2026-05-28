import copy
from functools import partial
from typing import Sequence, List

import torch
import torch.nn as nn
from ..modules.lora import LinearWithLoRA, LinearWithLoRA_QKVFused, LinearWithLoRA_KVFused
from .vit_lora_utils import Hooker, find_all_frozen_nn_linear_names, apply_lora


def apply_multi_lora(hooker: Hooker, model: nn.Module,
                     lora_module_names: List[str],
                     lora_num_replica: int, lora_enable_bits: Sequence[bool] | None,
                     merge_disabled_lora: bool,
                     lora_r: int, lora_alpha: float,
                     lora_dropout: float = 0.0, use_rslora: bool = False):
    for lora_module_name in lora_module_names:
        tokens = lora_module_name.split('.')
        parent_module_name = '.'.join(tokens[:-1])
        module_name = tokens[-1]
        parent_module = model.get_submodule(parent_module_name)

        if module_name == 'qkv':
            layer_cls = LinearWithLoRA_QKVFused
        elif module_name == 'kv':
            layer_cls = LinearWithLoRA_KVFused
        else:
            layer_cls = LinearWithLoRA

        if lora_enable_bits is not None:
            assert len(lora_enable_bits) == lora_num_replica

        linear = getattr(parent_module, module_name)
        replica = []
        for i in range(lora_num_replica):
            if lora_enable_bits is not None and not lora_enable_bits[i]:
                if merge_disabled_lora:
                    copied_linear = copy.deepcopy(linear)
                    if module_name == 'qkv':
                        copied_linear._register_load_state_dict_pre_hook(partial(hooker.qkv_linear_hook, copied_linear))
                    elif module_name == 'kv':
                        copied_linear._register_load_state_dict_pre_hook(partial(hooker.kv_linear_hook, copied_linear))
                    else:
                        copied_linear._register_load_state_dict_pre_hook(partial(hooker.linear_hook, copied_linear))
                    replica.append(copied_linear)
                else:
                    lora_obj = layer_cls(linear, lora_r, lora_alpha, lora_dropout, use_rslora)
                    for param in lora_obj.parameters():
                        param.requires_grad = False
                    replica.append(lora_obj)
            else:
                replica.append(layer_cls(linear, lora_r, lora_alpha, lora_dropout, use_rslora))
        replica = nn.ModuleList(replica)
        setattr(parent_module, module_name, replica)


def patch_model_with_multi_lora_(self: nn.Module, lora_r: int, num_replica: int, lora_enable_bits: Sequence[bool] | None,
                                 enable_mlp_moe: bool,
                                 merge_disabled_lora: bool):
    hooker = Hooker()

    for i_layer, block in enumerate(self.blocks):
        linear_names = find_all_frozen_nn_linear_names(block.attn)
        apply_multi_lora(hooker, block.attn, linear_names, num_replica, lora_enable_bits, merge_disabled_lora, lora_r, lora_r)

        linear_names = find_all_frozen_nn_linear_names(block.mlp)
        if enable_mlp_moe:
            apply_multi_lora(hooker, block.mlp, linear_names, num_replica, lora_enable_bits, merge_disabled_lora, lora_r, lora_r)
        else:
            apply_lora(block.mlp, linear_names, lora_r, lora_r)
    return self


def enable_model_with_multi_lora_inference_capable(model: nn.Module, num_replica: int):
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append(name)
    for linear_module_full_name in linear_modules:
        tokens = linear_module_full_name.split('.')
        parent_module_name = '.'.join(tokens[:-1])
        module_name = tokens[-1]

        parent_module = model.get_submodule(parent_module_name)
        linear_layer = getattr(parent_module, module_name)
        linear_layers = nn.ModuleList(copy.deepcopy(linear_layer) for _ in range(num_replica))
        setattr(parent_module, module_name, linear_layers)


def patch_model_with_multi_lora_weight_capable_(model: nn.Module,
                                                enable_mlp_moe: bool,
                                                num_replica: int, lora_alpha: None | float = None, use_rslora: bool = False):
    hooker = Hooker(lora_alpha, use_rslora)
    for i_layer, block in enumerate(model.blocks):
        enable_model_with_multi_lora_inference_capable(block.attn, num_replica)
        for name, module in block.attn.named_modules():
            if isinstance(module, nn.Linear):
                if 'qkv' in name:
                    module._register_load_state_dict_pre_hook(hooker.qkv_linear_hook, module)
                elif 'kv' in name:
                    module._register_load_state_dict_pre_hook(hooker.kv_linear_hook, module)
                else:
                    module._register_load_state_dict_pre_hook(hooker.linear_hook, module)

        if enable_mlp_moe:
            enable_model_with_multi_lora_inference_capable(block.mlp, num_replica)
        for name, module in block.mlp.named_modules():
            if isinstance(module, nn.Linear):
                if 'qkv' in name:
                    module._register_load_state_dict_pre_hook(hooker.qkv_linear_hook, module)
                elif 'kv' in name:
                    module._register_load_state_dict_pre_hook(hooker.kv_linear_hook, module)
                else:
                    module._register_load_state_dict_pre_hook(hooker.linear_hook, module)
