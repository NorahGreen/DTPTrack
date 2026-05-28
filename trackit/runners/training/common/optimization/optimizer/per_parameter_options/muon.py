import torch.nn as nn
from typing import Dict, Tuple


def apply_muon_rule_(rule: dict, module_parameters: Dict[str, nn.Parameter],
                                           optimizer_param_dict: list, decay_parameter_names: Tuple[str, ...]):
    muon_groups = []
    adamw_groups = []
    adamw_zero_weight_decay_groups = []
    for module_parameter_name in list(module_parameters.keys()):
        param = module_parameters.pop(module_parameter_name)
        if module_parameter_name not in decay_parameter_names or 'embed' in module_parameter_name:
            adamw_zero_weight_decay_groups.append(param)
        elif 'head' in module_parameter_name or 'final_layer' in module_parameter_name:
            adamw_groups.append(param)
        else:
            muon_groups.append(param)
    if len(muon_groups) > 0:
        optimizer_param_dict.append({'params': tuple(muon_groups), 'use_muon': True})
    if len(adamw_groups) > 0:
        optimizer_param_dict.append({'params': tuple(adamw_groups), 'use_muon': False})
    if len(adamw_zero_weight_decay_groups) > 0:
        optimizer_param_dict.append({'params': tuple(adamw_zero_weight_decay_groups), 'use_muon': False, 'weight_decay': 0.})
