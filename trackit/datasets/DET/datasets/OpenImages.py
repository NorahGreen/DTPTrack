from trackit.datasets.common.seed import BaseSeed


class OpenImages_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=('train', 'val', 'test')):
        if root_path is None:
            root_path = self.get_path_from_config('Open_Images_PATH')
        super().__init__('Open-Images-V6', root_path, data_split, ('train', 'val', 'test'))

    def construct(self, constructor):
        from .impl.OpenImages import construct_OpenImages
        construct_OpenImages(constructor, self)
