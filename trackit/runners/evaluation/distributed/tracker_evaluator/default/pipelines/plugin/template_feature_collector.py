import torch

from trackit.data.protocol.eval_input import TrackerEvalData
from ....components.tensor_cache import CacheService, TensorCache
from . import TrackingPipelinePlugin


class TemplateFeatureCollector(TrackingPipelinePlugin):
    def start(self, cache_batch_size: int, *_):
        self._max_batch_size = cache_batch_size

    def stop(self, *_):
        del self._max_batch_size
        if hasattr(self, 'template_feature_cache'):
            del self.template_feature_cache

    def on_initialized(self, data: TrackerEvalData, model_outputs, *_):
        if model_outputs is None:
            return

        task_ids = []
        for task in data.tasks:
            if task.tracker_do_init_context is not None:
                task_ids.append(task.id)

        z_feat = model_outputs['z_feat']

        if not hasattr(self, 'template_feature_cache'):
            self.template_feature_cache = _initialize(self._max_batch_size, z_feat.shape[1:],
                                                      z_feat.device, z_feat.dtype)

        self.template_feature_cache.put_batch(task_ids, z_feat)

    def prepare_tracking(self, data: TrackerEvalData, model_input_params: dict, *_):
        task_ids = []
        for task in data.tasks:
            if task.tracker_do_tracking_context is not None:
                task_ids.append(task.id)

        if len(task_ids) > 0:
            model_input_params['z_feat'] = self.template_feature_cache.get_batch(task_ids)

    def on_tracked(self, data: TrackerEvalData, *_):
        if not hasattr(self, 'template_feature_cache'):
            return
        for task in data.tasks:
            if task.do_task_finalization:
                self.template_feature_cache.delete(task.id)


def _initialize(max_batch_size, template_feature_shape, device: torch.device, dtype: torch.dtype):
    return CacheService(TensorCache(max_batch_size, template_feature_shape, device, dtype))
