from typing import Tuple, Any
import numpy as np
import torch

from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.data.protocol.eval_input import TrackerEvalData
from trackit.runners.evaluation.distributed.tracker_evaluator.default.types import TrackerEvaluationPipeline_Context
from ...plugin import TrackingPipelinePlugin
from .. import select_memory_frames


class TemplateFeatForegroundMaskGenerationAllTemplate(TrackingPipelinePlugin):
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int],
                 num_templates: int,
                 device: torch.device):
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.num_templates = num_templates
        self.device = device
        self.background_value = 0
        self.foreground_value = 1

    def start(self, *_):
        self.memory_mask_contexts = {}

    def stop(self, *_):
        assert len(self.memory_mask_contexts) == 0, "Memory masks should be cleared before stopping the plugin."
        del self.memory_mask_contexts

    def prepare_initialization(self, data: TrackerEvalData, model_input_params: dict, *_):
        for task in data.tasks:
            if task.tracker_do_init_context is not None:
                current_init_context = task.tracker_do_init_context
                if task.id not in self.memory_mask_contexts:
                    self.memory_mask_contexts[task.id] = []
                self.memory_mask_contexts[task.id].append((current_init_context.gt_bbox, current_init_context.input_data['curation_parameter']))

    def prepare_tracking(self, data: TrackerEvalData, model_input_params: dict, *_):
        template_masks = [[] for _ in range(self.num_templates)]
        for task in data.tasks:
            if task.tracker_do_tracking_context is not None:
                tracking_context = task.tracker_do_tracking_context
                frame_indices = select_memory_frames(tracking_context.frame_index, self.num_templates)
                for i, idx in enumerate(frame_indices):
                    bbox, cropping_params = self.memory_mask_contexts[task.id][idx]
                    template_mask = _generate_mask(bbox, cropping_params, self.template_size, self.template_feat_size,
                                                   self.background_value, self.foreground_value)
                    template_masks[i].append(template_mask)

        if len(template_masks[0]) > 0:
            for i in range(self.num_templates):
                model_input_params[f'z_{i}_feat_mask'] = torch.stack(template_masks[i], dim=0).to(self.device)

    def on_tracked(self, data: TrackerEvalData, model_outputs: Any, context: TrackerEvaluationPipeline_Context, *_):
        if 'memory_new_z_curated' in context.temporary_objects:
            for task_id in context.temporary_objects['memory_new_z_curated'].keys():
                bbox = context.temporary_objects['memory_new_z_curated_bbox'][task_id]
                if bbox is None:
                    self.memory_mask_contexts[task_id].append(self.memory_mask_contexts[task_id][-1])
                    continue
                cropping_params = context.temporary_objects['memory_new_z_curation_parameter'][task_id]
                self.memory_mask_contexts[task_id].append((bbox, cropping_params))

        for task in data.tasks:
            if task.do_task_finalization:
                del self.memory_mask_contexts[task.id]


def _generate_mask(bbox: np.ndarray, curation_parameter: np.ndarray,
                   template_size: Tuple[int, int], template_feat_size: Tuple[int, int],
                   background_value: int = 0, foreground_value: int = 1) -> torch.Tensor:
    template_mask = torch.full((template_feat_size[1], template_feat_size[0]), background_value, dtype=torch.long)
    stride = template_size[0] / template_feat_size[0], template_size[1] / template_feat_size[1]
    template_cropped_bbox = get_foreground_bounding_box(bbox, curation_parameter, np.array(template_size), stride)
    if bbox_is_valid(template_cropped_bbox):
        template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
        template_mask[template_cropped_bbox[1]: template_cropped_bbox[3], template_cropped_bbox[0]: template_cropped_bbox[2]] = foreground_value
    return template_mask
