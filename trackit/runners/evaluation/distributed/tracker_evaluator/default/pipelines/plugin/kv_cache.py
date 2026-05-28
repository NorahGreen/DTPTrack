from typing import Sequence

import torch

from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.miscellanies.system.machine.utils import sizeof_fmt
from trackit.runners.evaluation.distributed import EvaluatorContext
from . import TrackingPipelinePlugin
from ....components.tensor_cache import CacheService, MultiTensorCache, MultiTensorCache_ZeroCopy


class KVCacheMaintainer(TrackingPipelinePlugin):
    def __init__(self, zero_copy: bool = True, offload: bool = False):
        self.zero_copy = zero_copy
        self.offload = offload

    def start(self, context: EvaluatorContext, *_):
        max_num_concurrent_tasks = context.max_batch_size * context.num_input_data_streams
        self._max_num_concurrent_tasks = max_num_concurrent_tasks
        self._offload = self.offload

    def stop(self, *_):
        if hasattr(self, '_kv_cache'):
            del self._kv_cache
        del self._max_num_concurrent_tasks
        del self._offload

    def on_initialized(self, data: TrackerEvalData, model_outputs, *_):
        if model_outputs is None:
            return

        if 'kv_caches' in model_outputs and not hasattr(self, '_kv_cache'):
            num_layers = len(model_outputs['kv_caches'])
            layer_0_k, layer_0_v = model_outputs['kv_caches'][0]
            self._kv_cache = initialize_kv_cache(
                self._max_num_concurrent_tasks, num_layers,
                layer_0_k.shape[1:],
                layer_0_k.dtype, layer_0_k.device,
                self.zero_copy, self._offload)

        task_ids = []
        for task in data.tasks:
            if task.tracker_do_init_context is not None:
                task_ids.append(task.id)

        self._kv_cache.put_batch(task_ids, model_outputs['kv_caches'])

    def prepare_tracking(self, data: TrackerEvalData, model_input_params: dict, *_):
        if not hasattr(self, '_kv_cache'):
            return
        do_track_task_ids = []
        for task in data.tasks:
            if task.tracker_do_tracking_context is not None:
                do_track_task_ids.append(task.id)

        if len(do_track_task_ids) > 0:
            model_input_params['kv_caches'] = self._kv_cache.get_batch(do_track_task_ids)

    def on_tracked(self, data: TrackerEvalData, *_):
        if not hasattr(self, '_kv_cache'):
            return
        for task in data.tasks:
            if task.do_task_finalization:
                self._kv_cache.free(task.id)


def initialize_kv_cache(max_batch_size: int,
                        depth: int,
                        shape: Sequence[int],
                        dtype: torch.dtype, device: torch.device,
                        zero_copy: bool, offload: bool = False):
    # N, num_heads, L, head_dim
    all_layer_shape_list = tuple(shape for _ in range(depth))
    if zero_copy:
        assert not offload
        k_cache = CacheService(MultiTensorCache_ZeroCopy(max_batch_size, all_layer_shape_list, dtype))
        v_cache = CacheService(MultiTensorCache_ZeroCopy(max_batch_size, all_layer_shape_list, dtype))
        print(f'kv cache: initialized zero-copy KV cache. Estimated max occupied memory: {sizeof_fmt(k_cache.size_in_bytes() + v_cache.size_in_bytes())}.')
    else:
        if device.type != 'cuda' and offload:
            offload = False
        if device.type == 'cuda':
            if not offload:
                try:
                    k_cache = CacheService(MultiTensorCache(max_batch_size, all_layer_shape_list, device, dtype))
                    v_cache = CacheService(MultiTensorCache(max_batch_size, all_layer_shape_list, device, dtype))
                    print(f'kv cache: initialized KV cache, size {sizeof_fmt(k_cache.size_in_bytes() + v_cache.size_in_bytes())}, '
                          f'cache_batch_size: {max_batch_size}, depth: {depth}, shape: {shape}, device: {device}')
                except torch.cuda.OutOfMemoryError:
                    print(f'kv cache: Out of memory. Trying to allocate on local memory...')
                    k_cache = CacheService(MultiTensorCache(max_batch_size, all_layer_shape_list, torch.device('cpu'), dtype))
                    v_cache = CacheService(MultiTensorCache(max_batch_size, all_layer_shape_list, torch.device('cpu'), dtype))
                    offload = True
                    print(
                        f'kv cache: initialized KV cache (offloaded), size {sizeof_fmt(k_cache.size_in_bytes() + v_cache.size_in_bytes())}, '
                        f'cache_batch_size: {max_batch_size}, depth: {depth}, shape: {shape}')
            else:
                k_cache = CacheService(MultiTensorCache(max_batch_size, all_layer_shape_list, torch.device('cpu'),
                                                        dtype))
                v_cache = CacheService(MultiTensorCache(max_batch_size, all_layer_shape_list, torch.device('cpu'),
                                                        dtype))
                print(
                    f'kv cache: initialized KV cache (offloaded), size {sizeof_fmt(k_cache.size_in_bytes() + v_cache.size_in_bytes())}, '
                    f'cache_batch_size: {max_batch_size}, depth: {depth}, shape: {shape}')
        else:
            k_cache = CacheService(MultiTensorCache(max_batch_size, all_layer_shape_list, device, dtype))
            v_cache = CacheService(MultiTensorCache(max_batch_size, all_layer_shape_list, device, dtype))
            print(
                f'kv cache: initialized KV cache, size {sizeof_fmt(k_cache.size_in_bytes() + v_cache.size_in_bytes())}, '
                f'cache_batch_size: {max_batch_size}, depth: {depth}, shape: {shape}, device: {device}')
    return KVCache(k_cache, v_cache, device, offload)

class KVCache:
    def __init__(self, k_cache, v_cache, device, offload):
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.offload = offload
        self.device = device

    def put_batch(self, ids, kv_caches):
        k_caches = tuple(
            layer_kv_cache[0] if not self.offload else layer_kv_cache[0].cpu() for layer_kv_cache in kv_caches)
        v_caches = tuple(
            layer_kv_cache[1] if not self.offload else layer_kv_cache[1].cpu() for layer_kv_cache in kv_caches)

        self.k_cache.put_batch(ids, k_caches)
        self.v_cache.put_batch(ids, v_caches)

    def get_batch(self, ids):
        k_caches = self.k_cache.get_batch(ids)
        v_caches = self.v_cache.get_batch(ids)
        kv_caches = tuple(
            (k_cache, v_cache) if not self.offload else (k_cache.to(self.device), v_cache.to(self.device))
            for k_cache, v_cache in zip(k_caches, v_caches))
        return kv_caches

    def free(self, id_, allow_non_exists: bool=False):
        if allow_non_exists:
            try:
                self.k_cache.delete(id_)
                self.v_cache.delete(id_)
            except ValueError:
                pass
        else:
            self.k_cache.delete(id_)
            self.v_cache.delete(id_)

    def free_batch(self, ids, allow_non_exists: bool=False):
        if allow_non_exists:
            for id_ in ids:
                try:
                    self.k_cache.delete(id_)
                    self.v_cache.delete(id_)
                except ValueError:
                    pass
        else:
            for id_ in ids:
                self.k_cache.delete(id_)
                self.v_cache.delete(id_)

    def has(self, id_):
        return self.k_cache.has(id_)
