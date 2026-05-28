from trackit.datasets.common.seed import BaseSeed


class PTB_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('PTB_PATH')
        super().__init__('PTB', root_path)

    def construct(self, constructor):
        from .Impl.PTB import construct_PTB
        construct_PTB(constructor, self)
