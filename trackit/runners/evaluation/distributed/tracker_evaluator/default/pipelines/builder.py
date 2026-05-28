import torch


def build_tracker_evaluator_pipeline(pipeline_config: dict, config: dict, device: torch.device, num_epochs: int):
    if pipeline_config['type'] == 'DTPTrack':
        from .DTPTrack.builder import build_DTPTrack_pipeline
        return build_DTPTrack_pipeline(pipeline_config, config, device)
    else:
        raise NotImplementedError(pipeline_config['type'])
