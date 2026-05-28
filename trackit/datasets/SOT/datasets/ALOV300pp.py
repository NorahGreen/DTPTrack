from trackit.datasets.common.seed import BaseSeed


class ALOV300pp_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('ALOV300++_Frames_PATH')
        if annotation_path is None:
            self.annotation_path = self.get_path_from_config('ALOV300++_Annotation_PATH')
        super().__init__('ALOV300++', root_path)

    def construct(self, constructor):
        from .Impl.ALOV300pp import construct_ALOV300pp
        construct_ALOV300pp(constructor, self)
