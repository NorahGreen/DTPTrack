from trackit.datasets.common.seed import BaseSeed


class CrowdHuman_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=('train', 'val'), fine_anno_only=True):
        assert fine_anno_only
        name = 'CrowdHuman'
        if fine_anno_only:
            name += '_fine_anno_only'
        if root_path is None:
            root_path = self.get_path_from_config('CrowdHuman_PATH')
        super().__init__('CrowdHuman', root_path, data_split, ('train', 'val'))

    def construct(self, constructor):
        from .impl.CrowdHuman import construct_CrowdHuman
        construct_CrowdHuman(constructor, self)
