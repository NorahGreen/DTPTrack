from trackit.datasets.common.seed import BaseSeed


class LVIS_Seed(BaseSeed):
    def __init__(self, coco_2017_path=None, annotation_file_path=None, data_split=('train', 'val')):
        if coco_2017_path is None:
            coco_2017_path = self.get_path_from_config('COCO_2017_PATH')
        if annotation_file_path is None:
            annotation_file_path = self.get_path_from_config('LVIS_Annotation_PATH')
        self.lvis_annotation_file_path = annotation_file_path
        super().__init__('LVIS', coco_2017_path, data_split, ('train', 'val'))

    def construct(self, constructor):
        from .impl.LVIS import construct_LVIS
        return construct_LVIS(constructor, self)
