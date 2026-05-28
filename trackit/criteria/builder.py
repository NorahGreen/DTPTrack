import torch
import torch.nn as nn


def build_criterion(criterion_config: dict, device: torch.device, dtype: torch.dtype) -> nn.Module:
    if criterion_config['type'] == 'DTPTrack':
        from .methods.DTPTrack.builder import build_SPMTrack_criteria
        return build_SPMTrack_criteria(criterion_config)
    else:
        raise ValueError("unknown criterion type")
