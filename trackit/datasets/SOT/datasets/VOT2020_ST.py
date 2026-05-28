from trackit.datasets.common.seed import BaseSeed


class VOT2020_ST_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('VOT2020ST_PATH')
        super().__init__('VOT2020_ST', root_path)

    def construct(self, constructor):
        from .Impl.VOT_STB import construct_VOT_STB
        construct_VOT_STB(constructor, self)
