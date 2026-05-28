from trackit.datasets.common.seed import BaseSeed


class Objects365_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=('train', 'val')):
        if root_path is None:
            root_path = self.get_path_from_config('Objects365_PATH')
        super().__init__('Objects365', root_path, data_split, ('train', 'val'))

    def construct(self, constructor):
        from .impl.Objects365 import construct_Objects365
        construct_Objects365(constructor, self)
