# https://www.trace.ethz.ch/publications/2017/pathtrack/index.html
from trackit.datasets.common.seed import BaseSeed


class PathTrack_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=('train', 'val')):
        if root_path is None:
            root_path = self.get_path_from_config('PathTrack_PATH')
        super().__init__('PathTrack', root_path, data_split, ('train', 'val'))

    def construct(self, constructor):
        from .Impl.PathTrack import construct_PathTrack
        construct_PathTrack(constructor, self)
