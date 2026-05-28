import torch
import torch.nn as nn
import numpy as np
from trackit.core.utils.box_mask_convert import rect_from_mask
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params, apply_siamfc_cropping, \
    apply_siamfc_cropping_subpixel, reverse_siamfc_cropping_params
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform


class _Handler:
    def __init__(self, video_segmentation_model: nn.Module,
                 task_id: int, num_total_frames: int,
                 init_frame: torch.Tensor,
                 init_mask: torch.Tensor,
                 search_region_size: np.ndarray,
                 area_factor: float,
                 interpolation_mode: str,
                 norm_stat: str):
        self.area_factor = area_factor
        self.search_region_size = search_region_size
        self.interpolation_mode = interpolation_mode
        self.video_segmentation_model = video_segmentation_model
        self.task_id = task_id
        self.num_total_frames = num_total_frames
        self.normalization_ = get_dataset_norm_stats_transform(norm_stat, inplace=True)

        C, H, W = init_frame.shape
        assert C == 3, 'image must have 3 channels'
        assert init_mask.shape == (H, W), 'mask must have the same shape as the image'
        init_bbox = rect_from_mask(init_mask.cpu().numpy())
        curr_cropping_parameter = get_siamfc_cropping_params(init_bbox, area_factor, search_region_size)
        search_region, _, curr_cropping_parameter = \
            apply_siamfc_cropping(init_frame.to(torch.float32), search_region_size,
                                  curr_cropping_parameter,
                                  interpolation_mode)
        search_region.div_(255.0)
        self.normalization_(search_region)
        search_region_mask = apply_siamfc_cropping_subpixel(init_mask.to(torch.float32).unsqueeze(0).unsqueeze(0),
                                                            search_region_size,
                                                            curr_cropping_parameter,
                                                            interpolation_mode).squeeze(0).squeeze(0).to(torch.bool)
        self.search_region = search_region
        self.search_region_mask = search_region_mask
        self.frame_index = 0

    def init(self):
        self.video_segmentation_model(action='init', context={
            'frame': self.search_region, 'mask': self.search_region_mask,
            'total_num_frames': self.num_total_frames - self.frame_index, 'task_id': self.task_id,
            'frame_index': 0
        })

    def track(self, frame: torch.Tensor, predicted_bbox: np.ndarray, prediction_confidence: float):
        # do re-init here

        # start tracking
        self.frame_index += 1
        assert self.frame_index < self.num_total_frames, 'frame_index must be less than num_total_frames'
        C, H, W = frame.shape

        cropping_param = get_siamfc_cropping_params(predicted_bbox, self.area_factor, self.search_region_size)
        search_region, _, cropping_param = apply_siamfc_cropping(frame.to(torch.float32), self.search_region_size,
                                                                 cropping_param, self.interpolation_mode)
        search_region.div_(255.0)
        self.normalization_(search_region)

        results = self.video_segmentation_model(action='track', context={
            'frame': search_region, 'total_num_frames': self.num_total_frames - self.frame_index,
            'task_id': self.task_id, 'frame_index': self.frame_index
        })
        assert len(results) == 1 and self.task_id in results
        result = results[self.task_id]
        mask = result['mask'].sequeeze(0)

        full_mask = apply_siamfc_cropping_subpixel(mask.to(torch.float32).unsqueeze(0).unsqueeze(0),
                                                   np.array((W, H)),
                                                   reverse_siamfc_cropping_params(cropping_param),
                                                   self.interpolation_mode).squeeze(0).squeeze(0)
        return full_mask.to(torch.bool).cpu().numpy()

    def is_finished(self):
        return self.frame_index == self.num_total_frames - 1
