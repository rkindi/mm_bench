##################################################################################################

import os
import sys


import torch
# from triton.testing import do_bench

from atomic_file import PicklingResultCache
from io_utils import CSVProblemReader, hash_strings
from common_types import BiasType, GemmDataType, KDecomposition, TritonMMKernelStyle
from benchmark_base import BenchmarkBase
from torch_benchmarking_utils import prep_torch_inputs, do_bench
from parser_utils import parse_args

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CACHE_DIR = "tmp/eager_run/cache"

class EagerResultCache(PicklingResultCache):
    def __init__(self, cache_key: str):
        super().__init__(
            folder=f"{CACHE_DIR}/eager_profiler_logs", 
            cache_key=cache_key
        )

def get_eager_nullary(shape, torch_inputs):

    if shape.bias_type == BiasType.NONE:
        
        a, b, d = torch_inputs

        if a.dim() == 3:
            fn = torch.bmm
        elif a.dim() == 2:
            fn = torch.mm
        else:
            raise NotImplementedError

        nullary = lambda: fn(a, b, out=d)

    else:

        a, b, c, d = torch_inputs

        if a.dim() == 3:
            fn = torch.baddbmm
        elif a.dim() == 2:
            fn = torch.addmm
        else:
            raise NotImplementedError

        nullary = lambda: fn(c, a, b, out=d)
    
    return nullary


class EagerBenchmark(BenchmarkBase):
    def read(self):
        self.shapes = CSVProblemReader.read(self.parsed_args.in_file)

        self.shapes = list(
            filter(
                (lambda shape: shape.data_type == self.gemm_dtype), 
                self.shapes
            )
        )

    
    def get_compile_time_configs(self):
        pass
    
    def apply_pre_build_patches(self):
        pass
    
    def build(self):
        pass
    
    def _process_profiling_result(self, shape, result, gemm_dtype, k_decomp):
        cuda_version = os.environ["CUDA_VERSION"]
        assert len(result) == 1
        best_runtime = list(result.values())[0][0]
        b, m, n, k, layout = shape.b, shape.m, shape.n, shape.k, shape.layout.name
        backend = "EAGER"
        print(",".join(map(str, [backend,cuda_version,shape.op_type.name,shape.bias_type.name,gemm_dtype.name,k_decomp.name,"NONE",b,m,n,k,layout,best_runtime])))
    
    def profile_one(self, shape):
        
        result_cache = EagerResultCache(self.cache_key)

        if shape in result_cache:
            results = result_cache[shape]
            self._process_profiling_result(shape, results, self.gemm_dtype, self.k_decomp)
            return
        
        torch_inputs = prep_torch_inputs(shape, self.gemm_dtype)

        results = {}

        eager_nullary = get_eager_nullary(shape, torch_inputs)

        dur = do_bench(
            eager_nullary, 
            warmup=self.parsed_args.warmup_iterations, 
            rep=self.parsed_args.profiling_iterations
        )
        key = (shape,)
        results[key] = dur
        result_cache[shape] = results
        self._process_profiling_result(shape, results, self.gemm_dtype, self.k_decomp)
        sys.stdout.flush()

    def profile(self):
        cuda_version = os.environ["CUDA_VERSION"].replace(".", "")
        self.cache_key = f"d{self.gemm_dtype.name}_c{cuda_version}_w{self.parsed_args.warmup_iterations}_p{self.parsed_args.profiling_iterations}_t{self.parsed_args.trial_num}"
        for shape in self.shapes:
            self.profile_one(shape)

if __name__ == "__main__":

    parsed_args = parse_args(
        k_decomposition_choices = [KDecomposition.NONE], 
        kernel_style_choices = [TritonMMKernelStyle.NONE],
    )

    print("Backend,CUDA,Op Type,Bias Type,Data Type,K Decomposition,Kernel Style,B,M,N,K,Layout,Runtime")
    for gemm_data_type in GemmDataType:
        EagerBenchmark(parsed_args, gemm_data_type, KDecomposition.NONE).run()
    
