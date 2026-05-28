# https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html
from trackit.datasets.common.seed import BaseSeed


class SegTrackV2_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('SegTrackV2_PATH')
        super().__init__('SegTrackV2', root_path)

    def construct(self, constructor):
        from .Impl.SegTrackV2 import construct_SegTrackV2
        construct_SegTrackV2(constructor, self)
