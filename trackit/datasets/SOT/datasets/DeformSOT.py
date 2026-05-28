from trackit.datasets.common.seed import BaseSeed


class DeformSOT_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('DeformSOT_PATH')
        super().__init__('Deform-SOT', root_path)

    def construct(self, constructor):
        from .Impl.DeformSOT import construct_DeformSOT
        construct_DeformSOT(constructor, self)
