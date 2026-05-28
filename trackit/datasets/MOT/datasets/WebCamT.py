from trackit.datasets.common.seed import BaseSeed


class WebCamT_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=('train', 'val'), include_passenger: bool=False):
        if root_path is None:
            root_path = self.get_path_from_config('WebCamT_PATH')
        flags = []
        if not include_passenger:
            flags += 'exclude_passenger'
        super(WebCamT_Seed, self).__init__('WebCamT', root_path, data_split, ('train', 'val'),
                                           flags)
        self.include_passenger = include_passenger

    def construct(self, constructor):
        from .Impl.WebCamT import construct_WebCamT
        construct_WebCamT(constructor, self)
