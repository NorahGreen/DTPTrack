import torch

from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.miscellanies.system.machine.utils import sizeof_fmt
from trackit.runners.evaluation.distributed import EvaluatorContext
from . import TrackingPipelinePlugin
from ...types import TrackerEvaluationPipeline_Context
from ....components.tensor_cache import IndexAllocator
from trackit.models import ModelKVCacheSelfDescriptionMixin


class KVCacheMaintainer(TrackingPipelinePlugin):
    def __init__(self, device: torch.device):
        self._device = device

    def start(self, context: EvaluatorContext, global_objects: dict):
        kv_cache_dtype = context.auto_mixed_precision_dtype if context.auto_mixed_precision_dtype is not None else context.dtype
        max_concurrent_tasks = context.max_batch_size * context.num_input_data_streams
        assert isinstance(context.model, ModelKVCacheSelfDescriptionMixin)
        kv_cache_shapes = context.model.get_kv_cache_shapes(max_concurrent_tasks)
        self.kv_caches = tuple((torch.empty(shape, device=self._device, dtype=kv_cache_dtype), torch.empty(shape, device=self._device, dtype=kv_cache_dtype))
                               for shape in kv_cache_shapes)
        self._id_index_mapper = IndexAllocator(max_concurrent_tasks)
        print('kv cache: initialized KV cache, size: '
              f'{sizeof_fmt(sum(cache[0].element_size() * cache[0].nelement() * 2 for cache in self.kv_caches))}, '
              f'max_concurrent_tasks: {max_concurrent_tasks}, depth: {len(self.kv_caches)}, '
              f'device: {self._device}, dtype: {kv_cache_dtype}')

    def stop(self, context: EvaluatorContext, global_objects: dict):
        assert self._id_index_mapper.empty()
        del self._id_index_mapper
        del self.kv_caches

    def prepare_initialization(self, data: TrackerEvalData, model_input_params: dict,
                               context: TrackerEvaluationPipeline_Context):
        # do create context
        for task in data.tasks:
            if task.task_creation_context is not None:
                self._id_index_mapper.allocate(task.id)

        kv_cache_indices = []
        for task in data.tasks:
            if task.tracker_do_init_context is not None:
                kv_cache_indices.append(self._id_index_mapper.get_index(task.id))

        if len(kv_cache_indices) > 0:
            model_input_params['kv_caches'] = self.kv_caches
            model_input_params['kv_cache_batch_idx'] = torch.tensor(kv_cache_indices, dtype=torch.int32, device=self._device)

    def prepare_tracking(self, data: TrackerEvalData, model_input_params: dict, *_):
        kv_cache_indices = []
        for task in data.tasks:
            if task.tracker_do_tracking_context is not None:
                kv_cache_indices.append(self._id_index_mapper.get_index(task.id))

        if len(kv_cache_indices) > 0:
            model_input_params['kv_caches'] = self.kv_caches
            model_input_params['kv_cache_batch_idx'] = torch.tensor(kv_cache_indices, dtype=torch.int32, device=self._device)

    def on_tracked(self, data: TrackerEvalData, *_):
        for task in data.tasks:
            if task.do_task_finalization:
                self._id_index_mapper.free(task.id)
