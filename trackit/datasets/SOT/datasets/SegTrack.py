# http://cpl.cc.gatech.edu/projects/SegTrack/
from trackit.datasets.common.seed import BaseSeed


class SegTrack_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('SegTrack_PATH')
        super().__init__('SegTrack', root_path)

    def construct(self, constructor):
        from .Impl.SegTrack import construct_SegTrack
        construct_SegTrack(constructor, self)
