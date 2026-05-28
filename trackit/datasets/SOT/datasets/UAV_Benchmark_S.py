from trackit.datasets.common.seed import BaseSeed


class UAV_Benchmark_S_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_path: str = None):
        if root_path is None:
            root_path = self.get_path_from_config('UAVBenchmarkS_PATH')
        super().__init__('UAV-benchmark-S', root_path)
        self.annotation_path = annotation_path

    def construct(self, constructor):
        from .Impl.UAV_Benchmark_S import construct_UAVBenchmarkS
        construct_UAVBenchmarkS(constructor, self)
