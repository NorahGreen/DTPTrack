from typing import Sequence, Optional
from dataclasses import dataclass

from .. import SamplingResult_Element

@dataclass(frozen=True)
class SiameseTrainingPairSamplingResult:
    z: SamplingResult_Element
    x: SamplingResult_Element
    is_positive: bool
    aux_frames: Optional[Sequence[SamplingResult_Element]] = None
