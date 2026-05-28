from trackit.datasets.common.seed import BaseSeed


class DTB70_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('DTB70_PATH')
        super().__init__('DTB70', root_path)

    def construct(self, constructor):
        from .Impl.DTB70 import construct_DTB70
        construct_DTB70(constructor, self)
