from enum import Flag, auto
from trackit.datasets.common.seed import BaseSeed
from ..constructor import DetectionDatasetConstructor


class CityscapesAnnotationOption(Flag):
    PreferFine = auto()
    FineOnly = auto()
    CoarseOnly = auto()


class Cityscapes_Seed(BaseSeed):
    def __init__(self, root_path: str=None,
                 data_split=('train', 'val'),
                 annotation_option=CityscapesAnnotationOption.PreferFine, things_only=True):
        if root_path is None:
            root_path = self.get_path_from_config('Cityscapes_PATH')

        extra_flags = []
        if things_only:
            extra_flags.append('things_only')
        if annotation_option == CityscapesAnnotationOption.PreferFine:
            extra_flags.append('prefer_fine')
        elif annotation_option == CityscapesAnnotationOption.FineOnly:
            extra_flags.append('fine_only')
        elif annotation_option == CityscapesAnnotationOption.CoarseOnly:
            extra_flags.append('coarse_only')
        else:
            raise Exception
        super().__init__('Cityscapes', root_path, data_split, ('train', 'val'), extra_flags)
        self.annotation_option = annotation_option
        self.things_only = things_only

    def construct(self, constructor: DetectionDatasetConstructor):
        from .impl.Cityscapes import construct_CityScapes
        construct_CityScapes(constructor, self)
