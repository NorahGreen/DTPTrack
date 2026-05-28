from trackit.datasets.common.seed import BaseSeed


class NUSPRO_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('NUSPRO_PATH')
        super().__init__('NUS-PRO', root_path)

    def construct(self, constructor):
        from .Impl.NUS_PRO import construct_NUSPRO
        construct_NUSPRO(constructor, self)
