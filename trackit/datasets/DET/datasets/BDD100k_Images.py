from trackit.datasets.common.seed import BaseSeed
import os


class BDD100k_Images_Seed(BaseSeed):
    def __init__(self, root_path: str = None, labels_path: str = None, data_split=('train', 'val')):
        if root_path is None:
            root_path = self.get_path_from_config('BDD100k_PATH')
        root_path = os.path.join(root_path, 'images', '100k')
        super().__init__('BDD100k-Images', root_path, data_split, ('train', 'val'))
        self.labels_path = labels_path

    def construct(self, constructor):
        from .impl.BDD100k_Images import construct_BDD100k_Images
        construct_BDD100k_Images(constructor, self)
