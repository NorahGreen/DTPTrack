from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingResult_Element:
    dataset_index: int
    sequence_index: int
    track_id: int
    frame_index: int
