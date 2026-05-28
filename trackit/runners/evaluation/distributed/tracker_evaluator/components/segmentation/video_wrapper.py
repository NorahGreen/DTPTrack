import torch
import numpy as np
from typing import Tuple

from .vid import _Handler


class Segmentify_VideoSegmentationModel_PostProcessor:
    def __init__(self, model_name: str,
                 search_region_size: Tuple[int, int],
                 area_factor: float,
                 interpolation_mode: str,
                 device: torch.device,
                 norm_stat: str):
        self.model_name = model_name
        self.search_region_size = np.array(search_region_size)
        self.area_factor = area_factor
        self.interpolation_mode = interpolation_mode
        self.device = device
        self.norm_stat = norm_stat

    def start(self):
        self.tracking_videos = {}
        from trackit.models.methods.sam2_track.model import SAM2Track
        self.video_segmentation_model = SAM2Track(self.model_name, True)
        self.video_segmentation_model.eval()
        self.video_segmentation_model.to(self.device)

    def stop(self):
        assert len(self.tracking_videos) == 0, 'some tracking videos are not finished'
        del self.video_segmentation_model

    def init(self, task_id: int, num_total_frames: int, init_frame: torch.Tensor, init_mask: torch.Tensor):
        self.tracking_videos[task_id] = _Handler(self.video_segmentation_model,
                                                 task_id, num_total_frames,
                                                 init_frame, init_mask,
                                                 self.search_region_size,
                                                 self.area_factor, self.interpolation_mode,
                                                 self.norm_stat)

    def track(self, task_id: int, frame: torch.Tensor, predicted_bbox: np.ndarray, prediction_confidence: float
              ) -> np.ndarray:
        assert task_id in self.tracking_videos, 'task_id must be initialized first'
        return self.tracking_videos[task_id].track(frame, predicted_bbox, prediction_confidence)

    def finish(self, task_id: int):
        assert task_id in self.tracking_videos, 'task_id must be initialized first'
        del self.tracking_videos[task_id]
