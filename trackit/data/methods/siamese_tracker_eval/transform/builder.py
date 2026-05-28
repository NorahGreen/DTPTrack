import torch

from trackit.miscellanies.printing import pretty_format


def build_data_transform(transform_config: dict, config: dict,
                         device: torch.device = torch.device('cpu'),
                         dtype: torch.dtype = torch.float32):
    common_config = config['common']
    print('transform config:\n' + pretty_format(transform_config, indent_level=1))
    if transform_config['type'] == 'default':
        from .default import SiameseTrackerEval_DefaultDataTransform
        # currently only one type of transform is supported
        if 'template_sizes' in common_config and len(common_config['template_sizes']) == 1:
            template_size = common_config['template_sizes'][0]
        else:
            template_size = common_config['template_size']
        return SiameseTrackerEval_DefaultDataTransform(template_size,
                                                       transform_config['template_area_factor'],
                                                       transform_config.get('with_full_template_image', False),
                                                       common_config['interpolation_mode'],
                                                       common_config['interpolation_align_corners'],
                                                       common_config['normalization'],
                                                       device, dtype)
    elif transform_config['type'] == 'spmtrack':
        from .spmtrack import SiameseTrackerEval_DefaultDataTransform
        # currently only one type of transform is supported
        if 'template_sizes' in common_config and len(common_config['template_sizes']) == 1:
            template_size = common_config['template_sizes'][0]
        else:
            template_size = common_config['template_size']
        return SiameseTrackerEval_DefaultDataTransform(template_size,
                                                       transform_config['template_area_factor'],
                                                       transform_config.get('with_full_template_image', False),
                                                       common_config['interpolation_mode'],
                                                       common_config['interpolation_align_corners'],
                                                       common_config['normalization'],
                                                       device, dtype)
    elif transform_config['type'] == 'LoRATv2':
        from .LoRATv2 import SiameseTrackerEval_DefaultDataTransform
        # currently only one type of transform is supported
        if 'template_sizes' in common_config and len(common_config['template_sizes']) == 1:
            template_size = common_config['template_sizes'][0]
        else:
            template_size = common_config['template_size']
        return SiameseTrackerEval_DefaultDataTransform(template_size,
                                                       transform_config['template_area_factor'],
                                                       transform_config.get('with_full_template_image', False),
                                                       common_config['interpolation_mode'],
                                                       common_config['interpolation_align_corners'],
                                                       common_config['normalization'],
                                                       transform_config.get('template_area_factor_2', None),
                                                       device, dtype)
    elif transform_config['type'] == 'plain':
        from .plain import SiameseTrackerEval_PlainDataTransform
        return SiameseTrackerEval_PlainDataTransform(common_config['normalization'], device)
    else:
        raise ValueError(f"Unsupported transform type: {transform_config['type']}")
