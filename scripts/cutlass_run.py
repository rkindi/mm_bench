"""
v0 -- MVP

This script...

1. Takes a list of shapes for GEMM as input.
2. Determines the required compile-time configs.
3. Patches the CUTLASS profiler codegen scripts to use the desired configs.
4. Builds the CUTLASS profiler.
5. Produces each shape's runtime and FLOPS with the CUTLASS profiler as output.

"""

from multiprocessing import Pool as MultiProcessingPool
import os
import sys
import shutil
import subprocess
import tempfile
from typing import List, Tuple

from jinja2 import Template as JinjaTemplate

from atomic_file import ResultCache
from io_utils import CSVProblemReader, make_directories, hash_directory_contents, remove_directory_if_exists, hash_strings, Layout, replace_in_file
from mm_configs import get_mm_configs, get_cutlass_warp_counts
from common_types import GemmDataType, GemmFramework, KDecomposition, BiasType, TritonMMKernelStyle
from benchmark_base import BenchmarkBase
from parser_utils import parse_args

CUTLASS_DIR = "/cutlass"
# CUTLASS_CP_DIR = "tmp/cutlass_run/cutlass_cp"
CACHE_DIR = "tmp/cutlass_run/cache"

def _argsort(lst):
    return sorted(range(len(lst)),key=lst.__getitem__)

# https://github.com/NVIDIA/cutlass/blob/66d9cddc832c1cdc2b30a8755274f7f74640cfe6/tools/library/scripts/generator.py#L1792-L1887
GenerateSM80_TensorOp_16816_LINE_RANGE = (1792, 1887 + 1)
GenerateSM80_TensorOp_16816_TEMPLATE = JinjaTemplate(r'''#
def GenerateSM80_TensorOp_16816(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 16],                                    \
      DataType.{{dtype}}, DataType.{{dtype}}, DataType.f32,     \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [8, 4, 2]

  for math_inst in math_instructions:
    tile_descriptions = [
      {% for item in items %}
      TileDescription([{{", ".join([item.block_m|string(), item.block_n|string(), item.block_k|string()])}}], {{item.num_stages}}, {{get_cutlass_warp_counts(item)}}, math_inst, min_cc, max_cc),
      {% endfor %}
    ]

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints, \
        {% if stream_k %}
        swizzling_functor = SwizzlingFunctor.StreamK
        {% else %}
        swizzling_functor = SwizzlingFunctor.Identity8
        {% endif %}
      )
#
''')

# https://github.com/NVIDIA/cutlass/blob/66d9cddc832c1cdc2b30a8755274f7f74640cfe6/tools/library/scripts/generator.py#L2533-L2599
GenerateSM80_TensorOp_1688_fast_math_LINE_RANGE = (2533, 2599 + 1)
GenerateSM80_TensorOp_1688_fast_math_TEMPLATE = JinjaTemplate(r'''#
def GenerateSM80_TensorOp_1688_fast_math(manifest, cuda_version):

  if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
    return

  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  math_instructions = [
    MathInstruction(                                      \
      [16, 8, 8],                                         \
      DataType.tf32, DataType.tf32, DataType.f32,     \
      OpcodeClass.TensorOp,                               \
      MathOperation.multiply_add),
  ]

  min_cc = 80
  max_cc = 1024

  alignment_constraints = [4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      {% for item in items %}
      TileDescription([{{", ".join([item.block_m|string(), item.block_n|string(), item.block_k|string()])}}], {{item.num_stages}}, {{get_cutlass_warp_counts(item)}}, math_inst, min_cc, max_cc),
      {% endfor %}
    ]

    data_type = [DataType.f32, DataType.f32, DataType.f32, DataType.f32]

    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints, \
      {% if stream_k %}
      swizzling_functor = SwizzlingFunctor.StreamK
      {% else %}
      swizzling_functor = SwizzlingFunctor.Identity8
      {% endif %}
    )
#
    ''')

# https://github.com/NVIDIA/cutlass/blob/66d9cddc832c1cdc2b30a8755274f7f74640cfe6/tools/library/scripts/generator.py#L3919-L3958
GenerateSM80_LINE_RANGE = (3919, 3958 + 1)
GenerateSM80_TEMPLATE = JinjaTemplate(r'''#
def GenerateSM80(manifest, cuda_version):
  {{fn_name}}(manifest, cuda_version)
''')
                                      
# https://github.com/NVIDIA/cutlass/blob/66d9cddc832c1cdc2b30a8755274f7f74640cfe6/tools/library/scripts/generator.py#L4606-L4612
MAIN_BODY_LINE_RANGE = (4606, 4612 + 1)
MAIN_BODY_TEMPLATE = JinjaTemplate(r'''
  GenerateSM80(manifest, args.cuda_version)
''')


class PatchApplier:
    @staticmethod
    def _apply(
        source_text: str,
        line_ranges: List[Tuple[int, int]], # exclusive
        replacement_texts: List[str]
    ):
        # Map 1-based indexing to 0-based indexing. 1-based indexing is nicer for the interface
        # since we are patching files and text editors used to view those files tend to use 
        # 1-based indexing.
        line_ranges = [[line_range[0] - 1, line_range[1] - 1] for line_range in line_ranges]

        # Ensure non-overlapping ranges.
        if (
            len(set.union(*[set(range(line_range[0], line_range[1])) for line_range in line_ranges])) != 
            sum(line_range[1] - line_range[0] for line_range in line_ranges)
        ):
            raise ValueError(f"line_ranges ({line_ranges}) overlap.")

        # Sort ranges.
        argsort = _argsort(line_ranges)
        line_ranges = [line_ranges[i] for i in argsort]
        replacement_texts = [replacement_texts[i] for i in argsort]

        # Replace.
        source_text_lines = source_text.splitlines()
        lines = []
        current_range_idx = 0
        source_line_num = 0
        while source_line_num < len(source_text_lines):
            if current_range_idx >= len(line_ranges) or source_line_num < line_ranges[current_range_idx][0]:
                lines.append(source_text_lines[source_line_num])
                source_line_num += 1
            elif source_line_num == line_ranges[current_range_idx][0]:
                lines.append(replacement_texts[current_range_idx])
                source_line_num += line_ranges[current_range_idx][1] - line_ranges[current_range_idx][0]
            else:
                current_range_idx += 1

        return "\n".join(lines)

    @staticmethod
    def apply(
        fname: str,
        line_ranges: List[Tuple[int, int]], # exclusive
        replacement_texts: List[str],
    ):
        with open(fname, "r") as f:
            source_text = f.read()
        with open(fname, "w") as f:
            res = PatchApplier._apply(source_text, line_ranges, replacement_texts)
            f.write(res)

    @staticmethod
    def patch(gemm_dtype: GemmDataType, k_decomp: KDecomposition, configs, cutlass_cp_dir: str):
        if gemm_dtype == GemmDataType.TF32:
            inner_fn_name = "GenerateSM80_TensorOp_1688_fast_math"
            inner_fn_line_range = GenerateSM80_TensorOp_1688_fast_math_LINE_RANGE
            inner_fn_template = GenerateSM80_TensorOp_1688_fast_math_TEMPLATE
        elif gemm_dtype in (GemmDataType.F16, GemmDataType.BF16):
            inner_fn_name = "GenerateSM80_TensorOp_16816"
            inner_fn_line_range = GenerateSM80_TensorOp_16816_LINE_RANGE
            inner_fn_template = GenerateSM80_TensorOp_16816_TEMPLATE
        else:
            raise NotImplementedError

        inner_fn_replacement_str = inner_fn_template.render(
            items=configs, # items=configs[:2],
            get_cutlass_warp_counts=(lambda *args, **kwargs: list(get_cutlass_warp_counts(*args, **kwargs))),
            dtype=gemm_dtype.name.lower(),
            stream_k=(k_decomp == KDecomposition.STREAM_K),
        )
        GenerateSM80_replacement_str = GenerateSM80_TEMPLATE.render(fn_name=inner_fn_name)
        main_body_replacement_str = MAIN_BODY_TEMPLATE.render()
        PatchApplier.apply(
            f"{cutlass_cp_dir}/tools/library/scripts/generator.py", 
            [inner_fn_line_range, GenerateSM80_LINE_RANGE, MAIN_BODY_LINE_RANGE], 
            [inner_fn_replacement_str, GenerateSM80_replacement_str, main_body_replacement_str]
        )
        # print(inner_fn_replacement_str)
        # print(GenerateSM80_replacement_str)


def _clone_cutlass_codebase(cutlass_cp_dir, *, cutlass_dir=CUTLASS_DIR):
    remove_directory_if_exists(cutlass_cp_dir)
    # print(f"Cloning CUTLASS from {cutlass_dir} to {cutlass_cp_dir} to avoid modifying original.")
    shutil.copytree(cutlass_dir, cutlass_cp_dir)

def _build_cutlass(cache_key, cutlass_cp_dir):
    silent = True

    if silent:
        redirect_string = " > /dev/null 2>&1"
    else:
        redirect_string = ""

    template = JinjaTemplate('''#!/bin/bash
cd {{ directory }}
mkdir -p build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_CUBLAS=OFF -DCUTLASS_ENABLE_CUDNN=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON -DCUTLASS_LIBRARY_KERNELS=tensorop*gemm {{redirect_string}}
make cutlass_profiler -j$(nproc --all) {{redirect_string}}
''')
    script = template.render(directory=cutlass_cp_dir, redirect_string=redirect_string)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(script)

    os.chmod(f.name, 0o755)
    os.system(f.name)
    os.unlink(f.name)

    orig_fname = f"{cutlass_cp_dir}/build/tools/profiler/cutlass_profiler"
    cached_fname = f"{CACHE_DIR}/cutlass_profiler_bins/{cache_key}/cutlass_profiler"
    make_directories(cached_fname)
    shutil.copyfile(orig_fname, cached_fname)
    os.chmod(cached_fname, 0o755)

    # Move .so file.
    orig_fname = f"{cutlass_cp_dir}/build/tools/library/libcutlass.so"
    cached_fname = f"{CACHE_DIR}/libcutlass_sos/{cache_key}/libcutlass.so"
    make_directories(cached_fname)
    shutil.copyfile(orig_fname, cached_fname)


def _get_best_op(shape, s, gemm_dtype, k_decomp):
    if s.count("CSV Results:") != 1:
        return
    split_lines = [line.split(",") for line in s[s.find("CSV Results:"):].splitlines()]
    header_cols = split_lines[2]
    # print(header_cols)
    split_lines = split_lines[3:]
    runtime_col_idx = header_cols.index("Runtime")
    operation_col_idx = header_cols.index("Operation")

    runtimes = [split_line[runtime_col_idx] for split_line in split_lines]
    runtimes = list(filter(lambda x: float(x) > 0, runtimes))
    # operations = [split_line[operation_col_idx] for split_line in split_lines]

    if len(runtimes) == 0:
        return

    b, m, n, k, layout = shape.b, shape.m, shape.n, shape.k, shape.layout.name

    best_idx = _argsort(runtimes)[0]
    cuda_version = os.environ["CUDA_VERSION"]
    # print(f"For shape (b, m, n, k) = ({b}, {m}, {n}, {k}) & layout = {layout}, operator {operations[best_idx]} performs best at runtime {runtimes[best_idx]} ms.")
    print(",".join(map(str, ["CUTLASS",cuda_version,shape.op_type.name,shape.bias_type.name,gemm_dtype.name,k_decomp.name,"NONE",b,m,n,k,layout,runtimes[best_idx]])))

def _print_header():
    print("Backend,CUDA,Op Type,Bias Type,Data Type,K Decomposition,Kernel Style,B,M,N,K,Layout,Runtime")

def _parse_cutlass_profiler_stdout(shape, s, gemm_dtype, k_decomp):
    # print(s)

    _get_best_op(shape, s, gemm_dtype, k_decomp)

    return s.strip()


class CUTLASSResultCache(ResultCache):
    def __init__(self, cache_key: str):
        super().__init__(
            folder=f"{CACHE_DIR}/cutlass_profiler_logs", 
            cache_key=cache_key
        )

def _get_profiler_layout(layout: Layout):
    letter_map = {"r": "n", "c": "t"}
    result = "".join(map(letter_map.__getitem__, reversed(layout.name.lower())))
    return result


class CUTLASSBenchmark(BenchmarkBase):
    def __init__(self, parsed_args, gemm_dtype: GemmDataType, k_decomp: KDecomposition, bias_type: BiasType):
        self.bias_type = bias_type
        super().__init__(parsed_args, gemm_dtype, k_decomp)

    def read(self):
        self.shapes = CSVProblemReader.read(self.parsed_args.in_file)

        filter_fns = [
            # bias_type filter
            (lambda shape: shape.bias_type == self.bias_type),
            # dtype filter
            (lambda shape: shape.data_type == self.gemm_dtype),
            # if split-k, then skip batched.
            (lambda shape: not (self.k_decomp.is_split_k() and shape.op_type.is_batched())),
        ]

        for filter_fn in filter_fns:
            self.shapes = list(filter(filter_fn, self.shapes))

        # print(self.shapes)
    
    def get_compile_time_configs(self):
        self.compile_time_configs = get_mm_configs(GemmFramework.BOTH, GemmFramework.CUTLASS, self.gemm_dtype)
        # print(self.compile_time_configs)
    
    def apply_pre_build_patches(self):
        self.tmp_dir = tempfile.mkdtemp()
        # print(self.tmp_dir)
        # exit(0)

        self.cutlass_cp_dir = f"{self.tmp_dir}/cutlass_cp"

        _clone_cutlass_codebase(cutlass_cp_dir=self.cutlass_cp_dir)
        PatchApplier.patch(self.gemm_dtype, self.k_decomp, self.compile_time_configs, cutlass_cp_dir=self.cutlass_cp_dir)

        if self.bias_type.is_vector():
            replace_in_file(
                f"{self.cutlass_cp_dir}/tools/profiler/src/gemm_operation_profiler.cu",
                "gemm_workspace_.configuration.ldc = problem_.ldc;",
                "gemm_workspace_.configuration.ldc = 0;"
            )
        
        if not self.bias_type.is_batched():
            replace_in_file(
                f"{self.cutlass_cp_dir}/tools/profiler/src/gemm_operation_profiler.cu",
                "gemm_workspace_.C->batch_stride()",
                "0"
            )
            replace_in_file(
                f"{self.cutlass_cp_dir}/tools/profiler/src/gemm_operation_profiler.cu",
                "gemm_workspace_.arguments.batch_stride_C = gemm_workspace_.Reference->batch_stride();",
                "gemm_workspace_.arguments.batch_stride_C = 0;"
            )

        cuda_version = os.environ["CUDA_VERSION"].replace(".", "")
        self.cache_key = hash_strings(
            [
                hash_directory_contents(self.cutlass_cp_dir), 
                str(self.k_decomp == KDecomposition.STREAM_K), 
                self.gemm_dtype.name, 
                self.bias_type.name,
                cuda_version,
            ]
        )
    
    def build(self):
        cached_fname = f"{CACHE_DIR}/cutlass_profiler_bins/{self.cache_key}/cutlass_profiler"
        cached_fname_2 = f"{CACHE_DIR}/libcutlass_sos/{self.cache_key}/libcutlass.so"
        
        if os.path.isfile(cached_fname) and os.path.isfile(cached_fname_2):
            # print("Skipping profiler build because found in cache.")
            pass
        else:
            # print("Building profiler...")
            _build_cutlass(self.cache_key, cutlass_cp_dir=self.cutlass_cp_dir)

    def profile(self):
        # print(self.shapes)
        
        result_cache = CUTLASSResultCache(f"{self.cache_key}_{self.k_decomp.name}_w{self.parsed_args.warmup_iterations}_p{self.parsed_args.profiling_iterations}_t{self.parsed_args.trial_num}")

        for shape in self.shapes:
            if shape in result_cache:
                # print(f"FETCHING {shape} from cache...")
                _parse_cutlass_profiler_stdout(shape, result_cache[shape], self.gemm_dtype, self.k_decomp)
                continue

            # Note, _get_profiler_layout transposes internally because cutlass profiler gemms produce column major outputs.
            b, m, n, k, layout = shape.b, shape.m, shape.n, shape.k, _get_profiler_layout(shape.layout)

            if self.gemm_dtype == GemmDataType.TF32:
                alignments = [4, 2, 1]
                kernels_filter_fstring = " --kernels=cutlass_tensorop_s1688tf32gemm_*_{layout}_align{alignment}"
            elif self.gemm_dtype in (GemmDataType.BF16, GemmDataType.F16):
                alignments = [8, 4, 2]
                kernels_filter_fstring = " --kernels=cutlass_tensorop_{dtype_lower}_s16816gemm_{dtype_lower}_*_{layout}_align{alignment}"
            else:
                raise NotImplementedError
            
            beta = 0 if self.bias_type is BiasType.NONE else 1

            # Note, m and n are transposed because cutlass profiler gemms produce column major outputs.
            prof_cmd_fstring = f"LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{CACHE_DIR}/libcutlass_sos/{self.cache_key} {CACHE_DIR}/cutlass_profiler_bins/{self.cache_key}/cutlass_profiler --batch_count={b} --m={n} --n={m} --k={k} --min_cc=80 --beta={beta} --warmup-iterations={self.parsed_args.warmup_iterations} --profiling-iterations={self.parsed_args.profiling_iterations} --sleep-duration=1"
        
            dtype_lower = self.gemm_dtype.name.lower()

            for alignment in alignments:
                kernels_filter = kernels_filter_fstring.format(dtype_lower=dtype_lower, layout=layout, alignment=alignment)
                prof_cmd = prof_cmd_fstring + kernels_filter
                if self.k_decomp.is_split_k():
                    prof_cmd += f" --split_k_slices=1:16:1 --split_k_mode={self.k_decomp.split_k_shortname()}"
                output = subprocess.check_output(prof_cmd, shell=True)
                result = _parse_cutlass_profiler_stdout(shape, output.decode(), self.gemm_dtype, self.k_decomp)
                if result:
                    result_cache[shape] = result
                    break
        
            sys.stdout.flush()

def mp_build_fn(args):
    CUTLASSBenchmark(*args).run_through_build()

if __name__ == "__main__":

    parsed_args = parse_args(
        k_decomposition_choices = [
            KDecomposition.NONE,
            KDecomposition.SPLIT_K_SERIAL,
            KDecomposition.SPLIT_K_PARALLEL,
            KDecomposition.SPLIT_K_SERIAL_PARALLEL,
            KDecomposition.STREAM_K,
        ], 
        kernel_style_choices = [TritonMMKernelStyle.NONE],
    )

    with MultiProcessingPool(32) as p:
        p.map(
            mp_build_fn,
            [
                (parsed_args, gemm_data_type, parsed_args.k_decomposition, bias_type)
                for gemm_data_type in GemmDataType
                for bias_type in BiasType
            ]
        )
        
    _print_header()
    for gemm_data_type in GemmDataType:
        for bias_type in BiasType:
            CUTLASSBenchmark(parsed_args, gemm_data_type, parsed_args.k_decomposition, bias_type).run()