from typing import Tuple

import torch

from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box, generate_soft_foreground_mask
from trackit.data.protocol.eval_input import TrackerEvalData

from .... import EvaluatorContext
from ....components.tensor_cache import CacheService, TensorCache

from . import TrackingPipelinePlugin


class TemplateFeatForegroundMaskGeneration(TrackingPipelinePlugin):
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int], device: torch.device,
                 provide_during_tracking: bool = True):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.stride = template_size[0] // template_feat_size[0], template_size[1] // template_feat_size[1]
        self.device = device
        self.background_value = 0
        self.foreground_value = 1
        self.provide_during_tracking = provide_during_tracking

    def start(self, context: EvaluatorContext, *_):
        max_num_concurrent = context.max_batch_size * context.num_input_data_streams
        self.template_mask_cache = CacheService(TensorCache(max_num_concurrent,
                                                            (self.template_feat_size[1],
                                                             self.template_feat_size[0]),
                                                            self.device, torch.float32))
        self.seq_infos = {}

    def stop(self, *_):
        del self.seq_infos
        del self.template_mask_cache

    def prepare_initialization(self, data: TrackerEvalData, model_input_params: dict, context):
        do_init_task_ids = []
        for task in data.tasks:
            if task.task_creation_context is not None:
                seq_info = task.task_creation_context
                self.seq_infos[task.id] = seq_info
        for task in data.tasks:
            if task.tracker_do_init_context is not None:
                current_init_context = task.tracker_do_init_context
                try:
                    soft_template_mask = generate_soft_foreground_mask(current_init_context.gt_bbox,
                                                                       current_init_context.input_data['curation_parameter'],
                                                                       self.template_size, self.stride)
                except Exception as e:
                    from ..one_stream.visualization import visualize_template
                    seq_info = self.seq_infos[task.id]

                    from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping_to_boxes
                    siamfc_cropped_bbox = apply_siamfc_cropping_to_boxes(current_init_context.gt_bbox, current_init_context.input_data['curation_parameter'])
                    visualize_template(seq_info.dataset_name, seq_info.sequence_name, current_init_context.frame_index,
                                        'imagenet', current_init_context.input_data['curated_image'],
                                        siamfc_cropped_bbox)
                    raise RuntimeError(f"Failed to generate soft template mask for task {task.id}\nseq_info: {self.seq_infos[task.id]}\nbbox{current_init_context.gt_bbox}cropping_param:{current_init_context.input_data['curation_parameter']}") from e
                soft_template_mask = torch.from_numpy(soft_template_mask)
                self.template_mask_cache.put(task.id, soft_template_mask.to(self.device))
                do_init_task_ids.append(task.id)
        if not self.provide_during_tracking:
            if len(do_init_task_ids) > 0:
                model_input_params['z_feat_mask'] = self.template_mask_cache.get_batch(do_init_task_ids)

    def prepare_tracking(self, data: TrackerEvalData, model_input_params: dict, *_):
        if self.provide_during_tracking:
            do_track_task_ids = []
            for task in data.tasks:
                if task.tracker_do_tracking_context is not None:
                    do_track_task_ids.append(task.id)

            if len(do_track_task_ids) > 0:
                model_input_params['z_feat_mask'] = self.template_mask_cache.get_batch(do_track_task_ids)

    def on_tracked(self, data: TrackerEvalData, *_):
        for task in data.tasks:
            if task.do_task_finalization:
                self.template_mask_cache.delete(task.id)
                del self.seq_infos[task.id]
