import os
from trackit.datasets.common.seed import BaseSeed


class VisDrone2019_DET_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split='train'):
        if root_path is None:
            if data_split == 'train':
                root_path = self.get_path_from_config('VisDrone_2019_DET_Train_PATH')
            elif data_split == 'val':
                root_path = self.get_path_from_config('VisDrone_2019_DET_Val_PATH')
            elif data_split == 'test':
                root_path = self.get_path_from_config('VisDrone_2019_DET_Test_PATH')
            else:
                raise ValueError('Unsupported data type')
        images_path = os.path.join(root_path, 'images')
        annotation_path = os.path.join(root_path, 'annotations')

        super().__init__('VisDrone-2019-DET', images_path, data_split, ('train', 'val', 'test'))
        self.annotation_path = annotation_path

    def construct(self, constructor):
        from .impl.VisDrone_2019_DET import construct_VisDrone2019_DET
        construct_VisDrone2019_DET(constructor, self)
