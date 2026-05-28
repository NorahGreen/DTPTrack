def build_transform(data_config, config, build_context, dtype):
    transform_config = data_config['transform']
    if transform_config['type'] == 'default':
        from .default.builder import build_SPMTrack_data_processing_components
        return build_SPMTrack_data_processing_components(transform_config, config, build_context, dtype)
    elif transform_config['type'] == 'default_2':
        from .default_2.builder import build_SPMTrack_data_processing_components
        return build_SPMTrack_data_processing_components(transform_config, config, build_context, dtype)
    elif transform_config['type'] == 'default_global':
        from .default_global.builder import build_SPMTrack_data_processing_components
        return build_SPMTrack_data_processing_components(transform_config, config, build_context, dtype)
    else:
        raise NotImplementedError('Unknown transform type: {}'.format(transform_config['type']))
