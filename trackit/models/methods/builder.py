from torch import nn
from trackit.models import ModelBuildContext


def create_model_build_context(config: dict) -> ModelBuildContext:
    if config['type'] == 'DTPTrack':
        from .DTPTrack.builder import get_LoRAT_build_context
        build_context = get_LoRAT_build_context(config)
    else:
        raise NotImplementedError(config['type'])
    if isinstance(build_context, nn.Module):
        model = build_context
        build_context = ModelBuildContext(lambda impl_suggestion: model.to(impl_suggestion.device, impl_suggestion.dtype),
                                          lambda impl_suggestion: model.__class__.__name__ + str(impl_suggestion.device) + str(impl_suggestion.dtype))
    return build_context
