from trackit.datasets.common.seed import BaseSeed
import os


class KITTI_Detection_Seed(BaseSeed):
    def __init__(self, root_path: str=None, exclude_dontcare: bool=True):
        if root_path is None:
            root_path = self.get_path_from_config('KITTI_Detection_PATH')
        images_root_path = os.path.join(root_path, 'training', 'image_2')
        super().__init__('KITTI-Detection', images_root_path,
                         extra_flags='no_dontcare' if exclude_dontcare else None)
        self.exclude_dontcare = exclude_dontcare

    def construct(self, constructor):
        from .impl.KITTI import construct_KITTI_Detection
        construct_KITTI_Detection(constructor, self)
