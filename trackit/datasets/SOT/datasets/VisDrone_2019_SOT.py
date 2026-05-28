import os
from trackit.datasets.common.seed import BaseSeed


class VisDrone2019_SOT_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split='train'):
        if root_path is None:
            if data_split == 'train':
                root_path = self.get_path_from_config('VisDrone_2019_SOT_Train_PATH')
            elif data_split == 'val':
                root_path = self.get_path_from_config('VisDrone_2019_SOT_Val_PATH')
            elif data_split == 'test':
                root_path = self.get_path_from_config('VisDrone_2019_SOT_Test_PATH')
            else:
                raise ValueError('Unsupported data type')
        sequences_path = os.path.join(root_path, 'sequences')
        annotations_path = os.path.join(root_path, 'annotations')
        super().__init__('VisDrone2019-SOT', sequences_path, data_split, ('train', 'val', 'test'))
        self.annotations_path = annotations_path

    def construct(self, constructor):
        from .Impl.VisDrone_2019_SOT import construct_VisDrone_2019_SOT
        construct_VisDrone_2019_SOT(constructor, self)
