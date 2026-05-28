from trackit.datasets.common.seed import BaseSeed


class YoutubeBB_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=('train', 'val'), train_csv_path=None, validation_csv_path=None):
        if root_path is None:
            root_path = self.get_path_from_config('Youtube_BB_PATH')
        super().__init__('Youtube-BB', root_path, data_split, ('train', 'val'))
        self.train_csv = train_csv_path
        self.validation_csv = validation_csv_path

    def construct(self, constructor):
        from .Impl.YoutubeBB import construct_YoutubeBB
        construct_YoutubeBB(constructor, self)
