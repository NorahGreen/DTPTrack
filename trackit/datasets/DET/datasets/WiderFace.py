from trackit.datasets.common.seed import BaseSeed


class WiderFace_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=('train', 'val')):
        if root_path is None:
            root_path = self.get_path_from_config('WiderFace_PATH')
        super().__init__('WiderFace', root_path, data_split, ('train', 'val'))

    def construct(self, constructor):
        from .impl.WiderFace import construct_WiderFace
        construct_WiderFace(constructor, self)
