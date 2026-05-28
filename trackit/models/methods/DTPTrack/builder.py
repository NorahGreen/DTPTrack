from trackit.miscellanies.torch.dtype import set_default_dtype
from trackit.models import ModelBuildContext, ModelImplementationSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.printing import pretty_format
from .funcs.vit_lora_utils import attach_lora_state_dict_hooks_, enable_lora_


def get_LoRAT_build_context(config: dict):
    print('LoRAT model config:\n' + pretty_format(config['model'], indent_level=1))
    return ModelBuildContext(lambda impl_advice: build_LoRAT_model(config, impl_advice),
                             lambda impl_advice: get_LoRAT_build_string(config, impl_advice))


def build_LoRAT_model(config: dict, model_impl_suggestions: ModelImplementationSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'], model_impl_suggestions.load_pretrained,
                              device=model_impl_suggestions.device, dtype=model_impl_suggestions.dtype)
    model_type = model_config['type']
    with model_impl_suggestions.device, set_default_dtype(model_impl_suggestions.dtype):
        if model_type == 'DTPTrack':
            from .DTPTrack import DTPTrack
            model = DTPTrack(backbone, common_config['template_feat_size'],
                                                common_config['search_region_feat_size'],
                                                model_config['with_cls_token'], model_config['with_reg_token'],
                                                model_config.get('enable_aux_output', True), model_impl_suggestions.optimize_for_inference,
                                                model_config.get('aux_output_multi_head', True),
                                                **_get_mlora_4_config(config),)
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported.")
        if _is_lora_enabled(config):
            lora_r = model_config['lora']['r']
            lora_alpha = model_config['lora']['alpha']
            lora_dropout = model_config['lora']['dropout']
            use_rslora = model_config['lora']['use_rslora']
            lora_init_method = model_config['lora'].get('init_method', 'bert')
            if model_impl_suggestions.optimize_for_inference:
                attach_lora_state_dict_hooks_(model, lora_alpha, use_rslora)
            else:
                if 'gla' in model_type or 'gdn' in model_type or 'infini' in model_type:
                    pass
                elif 'radio' in model_type:
                    from .funcs.vit_backbone_freeze import freeze_radio_backbone_
                    freeze_radio_backbone_(model)
                else:
                    from .funcs.vit_backbone_freeze import freeze_vit_backbone_
                    freeze_vit_backbone_(model)
                enable_lora_(model, lora_r, lora_alpha, lora_dropout, use_rslora, lora_init_method)
        elif _is_multi_lora_enabled(config):
            if model_impl_suggestions.optimize_for_inference:
                from .funcs.patch_multi_lora import patch_model_with_multi_lora_weight_capable_
                patch_model_with_multi_lora_weight_capable_(model,
                                                            model_config.get('enable_mlp_moe', True),
                                                            model_config['multi_lora']['num_replica'])
            else:
                from .funcs.patch_multi_lora import patch_model_with_multi_lora_
                from .funcs.vit_backbone_freeze import freeze_loratv2_
                freeze_loratv2_(model,
                                model_config['multi_lora'].get('freeze_norm', True),
                                model_config['multi_lora'].get('freeze_head', False))
                patch_model_with_multi_lora_(model, model_config['multi_lora']['r'],
                                             model_config['multi_lora']['num_replica'],
                                             model_config['multi_lora']['lora_enable_bits'],
                                             model_config.get('enable_mlp_moe', True),
                                             model_config['multi_lora'].get('merge_disabled_lora', False))
        elif _is_lora_moe_enabled(config):
            from .funcs.patch_lora_moe import enable_lora_moe_
            from .funcs.vit_backbone_freeze import freeze_vit_backbone_
            freeze_vit_backbone_(model)
            enable_lora_moe_(model,
                             model_config['lora_moe']['r'],
                             model_config['lora_moe']['num_experts'],
                             model_config['lora_moe']['top_k'],
                             model_config['lora_moe']['use_shared_expert'],
                             model_config['lora_moe'].get('load_balance_coeff', 1e-3))
        else:
            attach_lora_state_dict_hooks_(model)
    return model


def get_LoRAT_build_string(config: dict, model_impl_suggestions: ModelImplementationSuggestions):
    model_type = config['model']['type']
    build_string = 'LoRAT_' + model_type
    if _is_lora_enabled(config) or _is_multi_lora_enabled(config):
        build_string += '_lora'
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
    if model_type in ('DTPTrack'):
        build_string += '_temporal_mlora'
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_inference'
    build_string += '_' + str(model_impl_suggestions.dtype)
    build_string += '_' + str(model_impl_suggestions.device)
    if model_impl_suggestions.load_pretrained:
        build_string += '_pretrained'
    return build_string


def _is_lora_enabled(config: dict):
    lora_enable = 'lora' in config['model']
    if lora_enable:
        lora_enable = config['model']['lora'].get('enabled', True)
    return lora_enable


def _is_multi_lora_enabled(config: dict):
    lora_enable = 'multi_lora' in config['model']
    if lora_enable:
        lora_enable = config['model']['multi_lora'].get('enabled', True)
    return lora_enable


def _is_lora_moe_enabled(config: dict):
    lora_enable = 'lora_moe' in config['model']
    if lora_enable:
        lora_enable = config['model']['lora_moe'].get('enabled', True)
    return lora_enable


def _get_mlora_4_config(config: dict):
    mlora_config = config['model']['mlora']
    return {
        'lambda_num': mlora_config['lambda_num'],
        'r': mlora_config['r'],
        'lora_alpha': mlora_config['alpha'],
        'lora_dropout': mlora_config['dropout'],
        'logging': mlora_config.get('logging', False),
        'init_method': mlora_config.get('init_method', 'default'),
    }
