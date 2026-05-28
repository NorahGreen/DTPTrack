import torch

from trackit.miscellanies.printing import pretty_format
from trackit.runners.evaluation.common.siamfc_search_region_cropping_params_provider.builder import \
    build_siamfc_search_region_cropping_parameter_provider_factory
from . import SPMTrack_EvaluationPipeline
from ....components.post_process.builder import build_post_process
from ....components.segmentation.builder import build_segmentify_post_processor

def build_DTPTrack_pipeline(pipeline_config: dict, config: dict, device: torch.device):
    common_config = config['common']

    print('pipeline: SPM tracker')
    print(pretty_format(pipeline_config, indent_level=1))

    plugins = _build_plugins(pipeline_config['plugin'], pipeline_config['num_templates'], config, device)

    return SPMTrack_EvaluationPipeline(
        device, common_config['template_size'], common_config['search_region_size'],
        pipeline_config['num_templates'],
        pipeline_config['template_area_factor'],
        pipeline_config.get('update_criteria', 0.),
        build_siamfc_search_region_cropping_parameter_provider_factory(pipeline_config['search_region_cropping']),
        build_post_process(pipeline_config['post_process'], common_config, device),
        build_segmentify_post_processor(pipeline_config['segmentify'], common_config,
                                        device) if 'segmentify' in pipeline_config else None,
        pipeline_config.get('trust_initial_results', False),
        common_config['interpolation_mode'], common_config['interpolation_align_corners'],
        common_config['normalization'],
        pipeline_config.get('visualization', False), plugins)


def _build_plugins(plugins_config: list[dict], num_templates: int, config: dict, device: torch.device):
    plugins = []
    for plugin_config in plugins_config:
        if plugin_config['type'] == 'template_foreground_indicating_mask_generation':
            from .plugin.template_foreground_indicating_mask_generation import TemplateFeatForegroundMaskGeneration
            plugins.append(TemplateFeatForegroundMaskGeneration(config['common']['template_size'],
                                                                config['common']['template_feat_size'],
                                                                device))
        elif plugin_config['type'] == 'template_foreground_indicating_mask_generation_all_template':
            from .plugin.template_foreground_indicating_mask_generation_all_template import TemplateFeatForegroundMaskGenerationAllTemplate
            plugins.append(TemplateFeatForegroundMaskGenerationAllTemplate(config['common']['template_size'],
                                                                           config['common']['template_feat_size'],
                                                                           num_templates,
                                                                           device))
        else:
            raise ValueError('Unknown plugin type: {}'.format(plugin_config['type']))
    return plugins
