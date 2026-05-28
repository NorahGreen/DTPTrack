from ...common.seed import BaseSeed
import os


class CityPersons_Seed(BaseSeed):
    def __init__(self, cityscapes_path: str=None, annotation_path: str=None, data_split=('train', 'val'), things_only: bool = True):
        flags = []
        if things_only:
            flags.append('things_only')
        cityscapes_path = os.path.join(cityscapes_path, 'leftImg8bit')
        if cityscapes_path is None:
            cityscapes_path = self.get_path_from_config('Cityscapes_PATH')
        if annotation_path is None:
            annotation_path = self.get_path_from_config('CityPersons_Annotation_PATH')
        super().__init__('citypersons', cityscapes_path, data_split, ('train', 'val'), flags)
        self.annotation_path = annotation_path
        self.things_only = things_only

    def construct(self, constructor):
        from .impl.CityPersons import construct_cityPersons
        construct_cityPersons(constructor, self)
