from typing import Optional, Tuple
import numpy as np
import math

from trackit.core.utils.siamfc_cropping import apply_siamfc_cropping_to_boxes


def _rasterize_bbox(bbox: np.ndarray, image_size: Tuple[int, int],
                   num_bins: Tuple[int, int], crop_box: Tuple[float, float, float, float] = (0., 1., 0., 1.),
                   crop_box_normalized: bool = True, output_format: str = 'bbox'):
    if crop_box_normalized:
        crop_box = crop_box[0] * image_size[0], crop_box[1] * image_size[1], crop_box[2] * image_size[0], crop_box[3] * image_size[1]

    bbox = bbox.copy()
    bbox[:, 0] = (bbox[:, 0] - crop_box[0]) / (crop_box[2] - crop_box[0]) * num_bins[0]
    bbox[:, 1] = (bbox[:, 1] - crop_box[1]) / (crop_box[3] - crop_box[1]) * num_bins[1]
    bbox[:, 2] = (bbox[:, 2] - crop_box[0]) / (crop_box[2] - crop_box[0]) * num_bins[0]
    bbox[:, 3] = (bbox[:, 3] - crop_box[1]) / (crop_box[3] - crop_box[1]) * num_bins[1]

    if output_format == 'bbox':
        bbox[:, 0] = np.clip(bbox[:, 0], 0, num_bins[0] - 1)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, num_bins[1] - 1)
        bbox[:, 2] = np.clip(bbox[:, 2], 0, num_bins[0] - 1)
        bbox[:, 3] = np.clip(bbox[:, 3], 0, num_bins[1] - 1)
        return bbox.astype(np.int64)
    elif output_format == 'center':
        center = np.zeros((bbox.shape[0], 2), dtype=np.float64)
        center[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.
        center[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.
        center[:, 0] = np.clip(center[:, 0], 0, num_bins[0] - 1)
        center[:, 1] = np.clip(center[:, 1], 0, num_bins[1] - 1)
        return center.astype(np.int64)
    else:
        raise ValueError(output_format)


def _normalize_bbox(bbox: np.ndarray, image_size: Tuple[int, int],
                    crop_box: Tuple[float, float, float, float] = (0., 1., 0., 1.),
                    crop_box_normalized: bool = True, output_format: str = 'bbox'):
    if crop_box_normalized:
        crop_box = crop_box[0] * image_size[0], crop_box[1] * image_size[1], crop_box[2] * image_size[0], crop_box[3] * image_size[1]

    bbox = bbox.copy()
    bbox[:, 0] = (bbox[:, 0] - crop_box[0]) / (crop_box[2] - crop_box[0])
    bbox[:, 1] = (bbox[:, 1] - crop_box[1]) / (crop_box[3] - crop_box[1])
    bbox[:, 2] = (bbox[:, 2] - crop_box[0]) / (crop_box[2] - crop_box[0])
    bbox[:, 3] = (bbox[:, 3] - crop_box[1]) / (crop_box[3] - crop_box[1])

    if output_format == 'center':
        center = np.zeros((bbox.shape[0], 2), dtype=np.float64)
        center[0, :] = (bbox[:, 0] + bbox[:, 2]) / 2.
        center[1, :] = (bbox[:, 1] + bbox[:, 3]) / 2.
        center = np.clip(center, 0., 1.)
        return center
    else:
        bbox = np.clip(bbox, 0., 1.)
        return bbox


def _do_sample(length: int, sampling_interval: int, max_num_samples: int):
    indices = np.arange(length - sampling_interval, -1, -sampling_interval)
    indices = np.flip(indices, 0)
    indices = indices[-max_num_samples:]
    return indices


def _sample_trajectory(trajectory_bbox: np.ndarray, trajectory_object_existence_flag: Optional[np.ndarray],
                       max_num_samples: int, sampling_interval: int):
    assert trajectory_bbox.ndim == 2

    sampled_indices = _do_sample(trajectory_bbox.shape[0], sampling_interval, max_num_samples)

    trajectory_bbox = trajectory_bbox[sampled_indices]

    if trajectory_object_existence_flag is not None:
        trajectory_object_existence_flag = trajectory_object_existence_flag[sampled_indices]

    return trajectory_bbox, trajectory_object_existence_flag, sampled_indices


def _non_existence_object_handling_fill_with_nearest(
        sampled_trajectory_bbox: np.ndarray, sampled_trajectory_object_existence_flag: np.ndarray,
        sampled_indices: np.ndarray,
        trajectory_bbox: np.ndarray, trajectory_object_existence_flag: np.ndarray):
    updated_sampled_indices = sampled_indices.copy()
    updated_sampled_trajectory_bbox = sampled_trajectory_bbox.copy()
    object_existence_frame_indices = np.nonzero(trajectory_object_existence_flag)[0]
    if len(object_existence_frame_indices) == len(trajectory_object_existence_flag):
        return updated_sampled_trajectory_bbox, updated_sampled_indices
    if len(object_existence_frame_indices) == 0:
        raise ValueError('No object existence frame found')

    object_existence_frame_indices = np.nonzero(trajectory_object_existence_flag)[0]

    for i, exists in enumerate(sampled_trajectory_object_existence_flag):
        if not exists:
            frame_index = sampled_indices[i]
            nearest_object_existence_frame_index = object_existence_frame_indices[np.argmin(np.abs(object_existence_frame_indices - frame_index))]
            updated_sampled_trajectory_bbox[i, :] = trajectory_bbox[nearest_object_existence_frame_index, :]
            updated_sampled_indices[i] = nearest_object_existence_frame_index
    return updated_sampled_trajectory_bbox, updated_sampled_indices


class TrajectoryEncodingHelper_WithSiamFCCropping:
    def __init__(self, max_num_samples: int,
                 image_size: Tuple[int, int], num_bins: Tuple[int, int],
                 crop_box: Tuple[float, float, float, float] = (0., 0., 1., 1.),
                 crop_box_normalized: bool = True, output_type: str = 'bbox',
                 non_existence_object_handling_method: str = 'fill_with_nearest'):
        assert max_num_samples > 0
        assert image_size[0]
        assert num_bins[0] > 0 and num_bins[1] > 0
        assert crop_box[0] < crop_box[2] and crop_box[1] < crop_box[3]
        assert output_type in ('bbox', 'center')
        assert non_existence_object_handling_method in ('fill_with_nearest', 'ignore')

        self._max_num_samples = max_num_samples
        self._image_size = image_size
        self._num_bins = num_bins
        self._crop_box = crop_box
        self._crop_box_normalized = crop_box_normalized
        self._output_type = output_type
        self._non_existence_object_handling_method = non_existence_object_handling_method

    def __call__(self, trajectory_bbox: np.ndarray, trajectory_object_existence_flag: Optional[np.ndarray],
                 SiamFC_scale_and_translate_parameter: np.ndarray, sampling_interval: int) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        assert sampling_interval > 0
        trajectory_bbox = trajectory_bbox.astype(np.float64)
        sampled_trajectory_bbox, sampled_trajectory_object_existence_flag, sampled_indices = _sample_trajectory(
            trajectory_bbox, trajectory_object_existence_flag, self._max_num_samples, sampling_interval)
        if len(sampled_trajectory_bbox) == 0:
            return False, None, None
        if self._non_existence_object_handling_method == 'fill_with_nearest' and trajectory_object_existence_flag is not None and not np.all(trajectory_object_existence_flag):
            try:
                sampled_trajectory_bbox, sampled_indices = _non_existence_object_handling_fill_with_nearest(
                    sampled_trajectory_bbox, sampled_trajectory_object_existence_flag, sampled_indices,
                    trajectory_bbox, trajectory_object_existence_flag)
            except ValueError:
                return False, None, None
        sampled_trajectory_bbox = apply_siamfc_cropping_to_boxes(sampled_trajectory_bbox, SiamFC_scale_and_translate_parameter)
        return True, _rasterize_bbox(sampled_trajectory_bbox, self._image_size, self._num_bins,
                                     self._crop_box, self._crop_box_normalized,
                                     self._output_type), sampled_indices


class TrajectoryEncodingHelper_WithSiamFCCropping_Normalized:
    def __init__(self, max_num_samples: int,
                 image_size: Tuple[int, int],
                 crop_box: Tuple[float, float, float, float] = (0., 0., 1., 1.),
                 crop_box_normalized: bool = True, output_type: str = 'bbox',
                 non_existence_object_handling_method: str = 'fill_with_nearest'):
        assert max_num_samples > 0
        assert image_size[0]
        assert crop_box[0] < crop_box[2] and crop_box[1] < crop_box[3]
        assert output_type in ('bbox', 'center')
        assert non_existence_object_handling_method in ('fill_with_nearest', 'ignore')

        self._max_num_samples = max_num_samples
        self._image_size = image_size
        self._crop_box = crop_box
        self._crop_box_normalized = crop_box_normalized
        self._output_type = output_type
        self._non_existence_object_handling_method = non_existence_object_handling_method

    def __call__(self, trajectory_bbox: np.ndarray, trajectory_object_existence_flag: Optional[np.ndarray],
                 SiamFC_scale_and_translate_parameter: np.ndarray, sampling_interval: int) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        assert sampling_interval > 0
        trajectory_bbox = trajectory_bbox.astype(np.float64)
        sampled_trajectory_bbox, sampled_trajectory_object_existence_flag, sampled_indices = _sample_trajectory(
            trajectory_bbox, trajectory_object_existence_flag, self._max_num_samples, sampling_interval)
        if len(sampled_trajectory_bbox) == 0:
            return False, None, None
        if self._non_existence_object_handling_method == 'fill_with_nearest' and trajectory_object_existence_flag is not None and not np.all(trajectory_object_existence_flag):
            try:
                sampled_trajectory_bbox, sampled_indices = _non_existence_object_handling_fill_with_nearest(
                    sampled_trajectory_bbox, sampled_trajectory_object_existence_flag, sampled_indices,
                    trajectory_bbox, trajectory_object_existence_flag)
                assert np.all(np.isfinite(sampled_trajectory_bbox)), sampled_trajectory_bbox
            except ValueError:
                return False, None, None
        else:
            assert np.all(np.isfinite(sampled_trajectory_bbox)), sampled_trajectory_bbox
        assert np.all(np.isfinite(SiamFC_scale_and_translate_parameter)), SiamFC_scale_and_translate_parameter
        sampled_trajectory_bbox = apply_siamfc_cropping_to_boxes(sampled_trajectory_bbox, SiamFC_scale_and_translate_parameter)
        assert np.all(np.isfinite(sampled_trajectory_bbox)), sampled_trajectory_bbox
        return True, _normalize_bbox(sampled_trajectory_bbox, self._image_size,
                                     self._crop_box, self._crop_box_normalized,
                                     self._output_type), sampled_indices


class TrajectoryEncodingHelper_WithSiamFCCropping_FixedLength:
    def __init__(self, num_samples: int,
                 sampling_rate: float,  # hz
                 image_size: Tuple[int, int], num_bins: Tuple[int, int],
                 crop_box: Tuple[float, float, float, float] = (0., 0., 1., 1.),
                 crop_box_normalized: bool = True, output_type: str = 'bbox', output_format: str = 'time_last',
                 non_existence_object_handling_method: str = 'fill_with_nearest',
                 empty_record_handling_method: str = 'fill_with_no_op',
                 default_trajectory_sampling_rate: float = 30,  # hz
                 dynamic_sampling_interval: bool = True):
        self._trajectory_encoding_helper = TrajectoryEncodingHelper_WithSiamFCCropping(
            num_samples, image_size, num_bins, crop_box, crop_box_normalized, output_type,
            non_existence_object_handling_method
        )
        assert output_format in ('time_last', 'time_first')
        assert empty_record_handling_method in ('fill_with_no_op', 'copy_first_frame')
        self._num_samples = num_samples
        assert sampling_rate > 0.
        self._sampling_rate = sampling_rate
        self._empty_record_handling_method = empty_record_handling_method
        output_channel = 4 if output_type == 'bbox' else 2
        self._trajectory_encoding_shape = (num_samples, output_channel)
        self._default_trajectory_sampling_rate = default_trajectory_sampling_rate
        self._dynamic_sampling_interval = dynamic_sampling_interval
        self._output_format = output_format

    def __call__(self, trajectory_bbox: np.ndarray, trajectory_object_existence_flag: Optional[np.ndarray],
                 SiamFC_scale_and_translate_parameter: np.ndarray, fps: Optional[float]) -> np.ndarray:
        if len(trajectory_bbox) == 0:
            encoded_trajectory = np.zeros(self._trajectory_encoding_shape, dtype=np.int64)
        else:
            if fps is None or not self._dynamic_sampling_interval:
                fps = self._default_trajectory_sampling_rate
            assert math.isfinite(fps)
            sampling_interval = int(round(fps / self._sampling_rate))
            if sampling_interval < 1:
                sampling_interval = 1

            max_input_length = (self._num_samples + 1) * sampling_interval
            trajectory_bbox = trajectory_bbox[-max_input_length:, :]
            trajectory_bbox = trajectory_bbox.astype(np.float64)
            if trajectory_object_existence_flag is not None:
                trajectory_object_existence_flag = trajectory_object_existence_flag[-max_input_length:]
            is_success, encoded_trajectory, sampled_indices = self._trajectory_encoding_helper(trajectory_bbox, trajectory_object_existence_flag, SiamFC_scale_and_translate_parameter, sampling_interval)

            if is_success:
                assert encoded_trajectory.shape[0] > 0
                if encoded_trajectory.shape[0] != self._num_samples:
                    if self._empty_record_handling_method == 'fill_with_no_op':
                        encoded_trajectory = np.concatenate((
                            np.zeros((self._num_samples - encoded_trajectory.shape[0], encoded_trajectory.shape[1]),
                                     dtype=np.int64),
                            encoded_trajectory + 1), axis=0)
                    elif self._empty_record_handling_method == 'copy_first_frame':
                        encoded_trajectory = np.concatenate((
                            np.tile(encoded_trajectory[0: 1, :],
                                    (self._num_samples - encoded_trajectory.shape[0], 1)),
                            encoded_trajectory), axis=0)
                        encoded_trajectory = encoded_trajectory + 1
                else:
                    encoded_trajectory = encoded_trajectory + 1
            else:
                encoded_trajectory = np.zeros(self._trajectory_encoding_shape, dtype=np.int64)
        if self._output_format == 'time_last':
            encoded_trajectory = encoded_trajectory.transpose((1, 0))
        return encoded_trajectory


class TrajectoryEncodingHelper_WithSiamFCCropping_FixedLength_Normalized:
    def __init__(self, num_samples: int,
                 sampling_rate: float,  # hz
                 image_size: Tuple[int, int],
                 crop_box: Tuple[float, float, float, float] = (0., 0., 1., 1.),
                 crop_box_normalized: bool = True, output_type: str = 'bbox', output_format: str = 'time_last',
                 non_existence_object_handling_method: str = 'fill_with_nearest',
                 empty_record_handling_method: str = 'fill_with_no_op',
                 default_trajectory_sampling_rate: float = 30,  # hz
                 dynamic_sampling_interval: bool = True,
                 dtype: np.dtype = np.float32):
        self._trajectory_encoding_helper = TrajectoryEncodingHelper_WithSiamFCCropping_Normalized(
            num_samples, image_size, crop_box, crop_box_normalized, output_type,
            non_existence_object_handling_method
        )
        assert output_format in ('time_last', 'time_first')
        assert empty_record_handling_method in ('fill_with_no_op', 'copy_first_frame')
        self._num_samples = num_samples
        self._sampling_rate = sampling_rate
        self._empty_record_handling_method = empty_record_handling_method
        output_channel = 4 if output_type == 'bbox' else 2
        self._trajectory_encoding_shape = (num_samples, output_channel)
        self._default_trajectory_sampling_rate = default_trajectory_sampling_rate
        self._dynamic_sampling_interval = dynamic_sampling_interval
        self._output_format = output_format
        self._output_dtype = dtype

    def __call__(self, trajectory_bbox: np.ndarray, trajectory_object_existence_flag: Optional[np.ndarray],
                 SiamFC_scale_and_translate_parameter: np.ndarray, fps: Optional[float]) -> np.ndarray:
        if len(trajectory_bbox) == 0:
            encoded_trajectory = np.full(self._trajectory_encoding_shape, -1, dtype=self._output_dtype)
        else:
            if fps is None or not self._dynamic_sampling_interval:
                fps = self._default_trajectory_sampling_rate
            sampling_interval = int(round(fps / self._sampling_rate))
            if sampling_interval < 1:
                sampling_interval = 1

            max_input_length = (self._num_samples + 1) * sampling_interval
            trajectory_bbox = trajectory_bbox[-max_input_length:, :]
            trajectory_bbox = trajectory_bbox.astype(np.float64)
            if trajectory_object_existence_flag is not None:
                trajectory_object_existence_flag = trajectory_object_existence_flag[-max_input_length:]
            is_success, encoded_trajectory, sampled_indices = self._trajectory_encoding_helper(trajectory_bbox, trajectory_object_existence_flag, SiamFC_scale_and_translate_parameter, sampling_interval)
            if is_success:
                assert encoded_trajectory.shape[0] > 0
                assert np.all(np.isfinite(encoded_trajectory)), encoded_trajectory
                if encoded_trajectory.shape[0] != self._num_samples:
                    if self._empty_record_handling_method == 'fill_with_no_op':
                        encoded_trajectory = np.concatenate((
                            np.full((self._num_samples - encoded_trajectory.shape[0], encoded_trajectory.shape[1]),
                                    -1, dtype=np.float64),
                            encoded_trajectory), axis=0)
                    elif self._empty_record_handling_method == 'copy_first_frame':
                        encoded_trajectory = np.concatenate((
                            np.tile(encoded_trajectory[0: 1, :],
                                    (self._num_samples - encoded_trajectory.shape[0], 1)),
                            encoded_trajectory), axis=0)
            else:
                encoded_trajectory = np.full(self._trajectory_encoding_shape, -1, dtype=np.float64)
        if self._output_format == 'time_last':
            encoded_trajectory = encoded_trajectory.transpose((1, 0))
        return encoded_trajectory.astype(self._output_dtype)
