from common_types import GemmDataType, KDecomposition

class BenchmarkBase:
    def __init__(self, parsed_args, gemm_dtype: GemmDataType, k_decomp: KDecomposition):
        self.parsed_args = parsed_args
        self.gemm_dtype = gemm_dtype
        self.k_decomp = k_decomp
    
    def run_through_build(self):
        self.read()

        if len(self.shapes) == 0:
            return
    
        self.get_compile_time_configs()
        self.apply_pre_build_patches()
        self.build()

    def run(self):
        self.read()

        if len(self.shapes) == 0:
            return
    
        self.get_compile_time_configs()
        self.apply_pre_build_patches()
        self.build()
        self.profile()

    def read(self):
        """
        Reads a file containing shapes to profile.

        Raises:
            NotImplementedError: To be implemented by subclasses.
        """
        raise NotImplementedError

    def get_compile_time_configs(self):
        """
        Determines compile time configs needed to build profiler.

        Raises:
            NotImplementedError: To be implemented by subclasses.
        """
        raise NotImplementedError
    
    def apply_pre_build_patches(self):
        """
        Applies patched to codebase required prior to build.

        Raises:
            NotImplementedError: To be implemented by subclasses.
        """
        raise NotImplementedError

    def build(self):
        """
        Build profiler.

        Raises:
            NotImplementedError: To be implemented by subclasses.
        """
        raise NotImplementedError

    def profile(self):
        """
        Profile shapes and produce output.

        Raises:
            NotImplementedError: To be implemented by subclasses.
        """
        raise NotImplementedError
