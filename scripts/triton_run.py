##################################################################################################

import os

# Setup env vars as first thing so when torch is initialized, it reads from these vars. Also
# important for any forked torch subprocesses such as the torchinductor kernel compile pool.
def _set_triton_env_vars():
    os.environ["CUDA_HOME"] = "/usr/local/cuda/"
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["TRITON_LIBDEVICE_PATH"] = "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "96" # TODO: use nproc
    # Reset CPU affinity (https://stackoverflow.com/a/15641148/21405443). We do this because
    # when using the cresset images, we find that the CPU affinity is being set to a limited
    # number of CPUs (maybe for some CPU BLAS library). However, by limiting the CPU affinity,
    # it means AsyncCompile's ProcessPoolExecutor's processes will all have the same affinity
    # even though there are many other cores that can do the work. Resetting the affinity lets
    # us freely use the other cores. We must do this before AsyncCompile is imported as we fork
    # upon importing the torch._inductor.codecache module.
    # os.system("taskset -p 0xff %d >/dev/null 2>&1" % os.getpid())

_set_triton_env_vars()

from torch._inductor.codecache import AsyncCompile
import torch._inductor.config
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
from triton.compiler import CompiledKernel
# from triton.testing import do_bench

torch._inductor.config.disable_progress = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from triton_legacy_utils import get_cuda_max_shared

_cuda_max_shared = get_cuda_max_shared()

def _patched_wait(self, scope):
    from torch._inductor.codecache import Future, TritonFuture, _compile_end, _Faketqdm, tqdm
    import torch._inductor.config as config
    from triton.compiler import OutOfResources

    # print("_patched_wait")

    num_kernels = len(
        [
            value
            for key, value in scope.items()
            if isinstance(value, (Future, TritonFuture))
        ]
    )
    pbar = tqdm(
        total=num_kernels,
        desc="Inductor Compilation",
        disable=config.disable_progress,
        delay=0,
    )
    if config.compile_threads > 1:
        failed_compile_keys = []
        for key, result in scope.items():
            if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                pbar.set_postfix_str(str(key))
            if isinstance(result, (Future, TritonFuture)):
                try:
                    scope[key] = result.result()
                    for launcher in scope[key].launchers:
                        if hasattr(launcher, "shared") and launcher.shared > _cuda_max_shared:
                            raise OutOfResources(launcher.shared, _cuda_max_shared, "shared memory")
                except OutOfResources as e:
                    failed_compile_keys.append(key)
                pbar.update(1)
        
        # print("FAILED_COMPILE_KEYS", failed_compile_keys)
        for key in failed_compile_keys:
            del scope[key]

    _compile_end()


AsyncCompile.wait = _patched_wait

def _patched_init_handles(x):
    pass

# For compatability with triton legacy.
if not hasattr(CompiledKernel, "_init_handles"):
    CompiledKernel._init_handles = _patched_init_handles

##################################################################################################

from pprint import pprint
from enum import Enum
import math
import shutil
from functools import lru_cache
import subprocess
import tempfile
import sys
from typing import List, Tuple

from jinja2 import Template as JinjaTemplate

from atomic_file import PicklingResultCache
from io_utils import CSVProblemReader, make_directories, hash_directory_contents, remove_directory_if_exists, hash_strings
from mm_configs import get_mm_configs, get_cutlass_warp_counts
from common_types import GemmDataType, GemmFramework, KDecomposition, BiasType, TritonMMKernelStyle
from benchmark_base import BenchmarkBase
from torch_benchmarking_utils import rand_strided, gemm_dtype_to_torch_dtype, prep_torch_inputs, allclose, do_bench
from eager_run import get_eager_nullary
from parser_utils import parse_args

async_compile = AsyncCompile()

def _get_input_signature_ptr_type(gemm_dtype):
    d = {
        GemmDataType.BF16: "bf16",
        GemmDataType.F16: "fp16",
        GemmDataType.TF32: "fp32",
    }
    return d[gemm_dtype]


def _get_template_inductor_mm(shape, compile_time_config, gemm_dtype, split_k=1):
    assert split_k == 1, "Split-K != 1 not supported"
    template = JinjaTemplate(r'''
import triton.language as tl
import triton
from torch._inductor.triton_ops.autotune import template
from torch._inductor.utils import instance_descriptor

@template(
    num_stages={{num_stages}},
    num_warps={{num_warps}},
    meta={
        'signature': {
            0: '*{{in_out_ptr_type}}', 
            1: '*{{in_out_ptr_type}}', 
            2: '*{{in_out_ptr_type}}',
            {% if has_bias %}
            3: '*{{in_out_ptr_type}}',
            {% endif %}
        },
        'device': {{dev}},
        'constants': {},
        'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, {% if has_bias %} 3 {% endif %}), equal_to_1=())]
    }
)
@triton.jit
def triton_(
    arg_A, 
    arg_B, 
    {% if has_bias %}
    in_ptr2,
    {% endif %}
    out_ptr1,
):

    GROUP_M : tl.constexpr = {{group_m}}
    EVEN_K : tl.constexpr = {{even_k}}
    ALLOW_TF32 : tl.constexpr = {{allow_tf32}}
    ACC_TYPE : tl.constexpr = {{acc_type}}
    BLOCK_M : tl.constexpr = {{block_m}}
    BLOCK_N : tl.constexpr = {{block_n}}
    BLOCK_K : tl.constexpr = {{block_k}}

    A = arg_A
    B = arg_B

    M = {{m}}
    N = {{n}}
    K = {{k}}

    {% if layout.name.lower()[0] == "r" %}
    stride_am = {{k}}
    stride_ak = 1
    {% else %}
    stride_am = 1
    stride_ak = {{m}}
    {% endif %}

    {% if layout.name.lower()[1] == "r" %}
    stride_bk = {{n}}
    stride_bn = 1
    {% else %}
    stride_bk = 1
    stride_bn = {{k}}
    {% endif %}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + ({{n}}*idx_m)
{{suffix}}
    ''')
    
    if shape.bias_type is BiasType.NONE:
        suffix = r'''
    tl.store(out_ptr1 + (xindex + tl.zeros(mask.shape, tl.int32)), acc, mask)
'''
    elif shape.bias_type is BiasType.VECTOR:
        suffix = r'''
    tmp0 = tl.load(in_ptr2 + (idx_n + tl.zeros(mask.shape, tl.int32)), mask)
    tmp1 = acc + tmp0
    tl.store(out_ptr1 + (xindex + tl.zeros(mask.shape, tl.int32)), tmp1, mask)
'''
    elif shape.bias_type is BiasType.MATRIX:
        suffix = JinjaTemplate(r'''
    tmp0 = tl.load(in_ptr2 + (idx_n + ({{n}}*idx_m) + tl.zeros(mask.shape, tl.int32)), mask)
    tmp1 = acc + tmp0
    tl.store(out_ptr1 + (xindex + tl.zeros(mask.shape, tl.int32)), tmp1, mask)
''').render(n=shape.n)
    else:
        raise NotImplementedError

    return template.render(
        m=shape.m,
        n=shape.n,
        k=shape.k,
        num_stages=compile_time_config.num_stages,
        num_warps=compile_time_config.num_warps,
        dev=0,
        group_m=8,
        even_k=(shape.k % (compile_time_config.block_k * split_k) == 0),
        allow_tf32=True,
        acc_type="tl.float32",
        block_m=compile_time_config.block_m,
        block_n=compile_time_config.block_n,
        block_k=compile_time_config.block_k,
        layout=shape.layout,
        in_out_ptr_type=_get_input_signature_ptr_type(gemm_dtype),
        has_bias=not shape.bias_type.is_none(),
        suffix=suffix,
    )

def _get_template_inductor_bmm(shape, compile_time_config, gemm_dtype):
    template = JinjaTemplate(r'''
import triton.language as tl
import triton
from torch._inductor.triton_ops.autotune import template
from torch._inductor.utils import instance_descriptor

@template(
    num_stages={{num_stages}},
    num_warps={{num_warps}},
    meta={
        'signature': {
            0: '*{{in_out_ptr_type}}', 
            1: '*{{in_out_ptr_type}}', 
            2: '*{{in_out_ptr_type}}',
            {% if has_bias %}
            3: '*{{in_out_ptr_type}}',
            {% endif %}
        },
        'device': {{dev}},
        'constants': {},
        'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, {% if has_bias %} 3 {% endif %}), equal_to_1=())]
    }
)
@triton.jit
def triton_(
    arg_A, 
    arg_B, 
    {% if has_bias %}
    in_ptr2,
    {% endif %}
    out_ptr1,
):

    GROUP_M : tl.constexpr = {{group_m}}
    EVEN_K : tl.constexpr = {{even_k}}
    ALLOW_TF32 : tl.constexpr = {{allow_tf32}}
    ACC_TYPE : tl.constexpr = {{acc_type}}
    BLOCK_M : tl.constexpr = {{block_m}}
    BLOCK_N : tl.constexpr = {{block_n}}
    BLOCK_K : tl.constexpr = {{block_k}}

    A = arg_A
    B = arg_B

    M = {{m}}
    N = {{n}}
    K = {{k}}

    stride_aq = {{m * k}}
    {% if layout.name.lower()[0] == "r" %}
    stride_am = {{k}}
    stride_ak = 1
    {% else %}
    stride_am = 1
    stride_ak = {{m}}
    {% endif %}

    stride_bq = {{k * n}}
    {% if layout.name.lower()[1] == "r" %}
    stride_bk = {{n}}
    stride_bn = 1
    {% else %}
    stride_bk = 1
    stride_bn = {{k}}
    {% endif %}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + ({{n}}*idx_m) + ({{m * n}}*idx_q)
{{suffix}}
    ''')
    split_k = 1

    if shape.bias_type is BiasType.NONE:
        suffix = r'''
    tl.store(out_ptr1 + (xindex + tl.zeros(mask.shape, tl.int32)), acc, mask)
'''
    elif shape.bias_type is BiasType.VECTOR:
        suffix = r'''
    tmp0 = tl.load(in_ptr2 + (idx_n + tl.zeros(mask.shape, tl.int32)), mask)
    tmp1 = tmp0 + acc
    tl.store(out_ptr1 + (xindex + tl.zeros(mask.shape, tl.int32)), tmp1, mask)
'''
    elif shape.bias_type is BiasType.MATRIX:
        suffix = JinjaTemplate(r'''
    x4 = xindex % {{m * n}}
    tmp0 = tl.load(in_ptr2 + (x4 + tl.zeros(mask.shape, tl.int32)), mask)
    tmp1 = tmp0 + acc
    tl.store(out_ptr1 + (xindex + tl.zeros(mask.shape, tl.int32)), tmp1, mask)
''').render(m=shape.m, n=shape.n)
    elif shape.bias_type is BiasType.BATCHED_VECTOR:
        suffix = JinjaTemplate(r'''
    tmp0 = tl.load(in_ptr2 + (idx_n + ({{n}} * idx_q) + tl.zeros(mask.shape, tl.int32)), mask)
    tmp1 = tmp0 + acc
    tl.store(out_ptr1 + (idx_n + ({{n}} *idx_m) + ({{m * n}}*idx_q) + tl.zeros(mask.shape, tl.int32)), tmp1, mask)
''').render(m=shape.m, n=shape.n)
    elif shape.bias_type is BiasType.BATCHED_MATRIX:
        suffix = r'''
    tmp0 = tl.load(in_ptr2 + (xindex + tl.zeros(mask.shape, tl.int32)), mask)
    tmp1 = tmp0 + acc
    tl.store(out_ptr1 + (xindex + tl.zeros(mask.shape, tl.int32)), tmp1, mask)
'''
    else:
        raise NotImplementedError
    

    return template.render(
        m=shape.m,
        n=shape.n,
        k=shape.k,
        num_stages=compile_time_config.num_stages,
        num_warps=compile_time_config.num_warps,
        dev=0,
        group_m=8,
        even_k=(shape.k % (compile_time_config.block_k * split_k) == 0),
        allow_tf32=True,
        acc_type="tl.float32",
        block_m=compile_time_config.block_m,
        block_n=compile_time_config.block_n,
        block_k=compile_time_config.block_k,
        layout=shape.layout,
        in_out_ptr_type=_get_input_signature_ptr_type(gemm_dtype),
        has_bias=not shape.bias_type.is_none(),
        suffix=suffix,
    )

def _get_template_triton_ops_matmul(shape, compile_time_config, gemm_dtype, split_k=1):
    template = JinjaTemplate(r'''
import triton.language as tl
import triton
from torch._inductor.triton_ops.autotune import template
from torch._inductor.utils import instance_descriptor

@template(
    num_stages={{num_stages}},
    num_warps={{num_warps}},
    meta={
        'signature': {0: '*{{in_out_ptr_type}}', 1: '*{{in_out_ptr_type}}', 2: '*{{in_out_ptr_type}}'},
        'device': {{dev}},
        'constants': {},
        'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]
    }
)
@triton.jit
def triton_(A, B, C):

    GROUP_M : tl.constexpr = {{group_m}}
    EVEN_K : tl.constexpr = {{even_k}}
    ALLOW_TF32 : tl.constexpr = {{allow_tf32}}
    ACC_TYPE : tl.constexpr = {{acc_type}}
    BLOCK_M : tl.constexpr = {{block_m}}
    BLOCK_N : tl.constexpr = {{block_n}}
    BLOCK_K : tl.constexpr = {{block_k}}
    SPLIT_K : tl.constexpr = {{split_k}}

    M = {{m}}
    N = {{n}}
    K = {{k}}

    {% if layout.name.lower()[0] == "r" %}
    stride_am = {{k}}
    stride_ak = 1
    {% else %}
    stride_am = 1
    stride_ak = {{m}}
    {% endif %}

    {% if layout.name.lower()[1] == "r" %}
    stride_bk = {{n}}
    stride_bn = 1
    {% else %}
    stride_bk = 1
    stride_bn = {{k}}
    {% endif %}

    # row-major (n)
    stride_cm = {{n}}
    stride_cn = 1

    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)
    ''')
    return template.render(
        m=shape.m,
        n=shape.n,
        k=shape.k,
        num_stages=compile_time_config.num_stages,
        num_warps=compile_time_config.num_warps,
        dev=0,
        group_m=8,
        even_k=(shape.k % (compile_time_config.block_k * split_k) == 0),
        allow_tf32=True,
        acc_type="tl.float32",
        block_m=compile_time_config.block_m,
        block_n=compile_time_config.block_n,
        block_k=compile_time_config.block_k,
        split_k=split_k,
        layout=shape.layout,
        in_out_ptr_type=_get_input_signature_ptr_type(gemm_dtype),
    )


def _get_template_bmm(kernel_style: TritonMMKernelStyle, shape, compile_time_config, gemm_dtype):
    return _get_template_inductor_bmm(shape, compile_time_config, gemm_dtype)

def _get_template_mm(kernel_style: TritonMMKernelStyle, shape, compile_time_config, gemm_dtype, split_k=1):
    if kernel_style == TritonMMKernelStyle.INDUCTOR:
        fn = _get_template_inductor_mm
    elif kernel_style == TritonMMKernelStyle.TRITON_OPS_MATMUL:
        fn = _get_template_triton_ops_matmul
    else:
        raise NotImplementedError
    return fn(shape, compile_time_config, gemm_dtype, split_k=split_k)

def _get_template(triton_mm_kernel_style, shape, compile_time_config, gemm_dtype, split_k=1):
    if shape.b == 1:
        return _get_template_mm(triton_mm_kernel_style, shape, compile_time_config, gemm_dtype, split_k=split_k)
    else:
        return _get_template_bmm(triton_mm_kernel_style, shape, compile_time_config, gemm_dtype)

def cdiv(a, b):
    return math.ceil(a / b)

@lru_cache(maxsize=None)
def get_cuda_stream_0():
    return get_cuda_stream(0)

def get_wrapped_triton_op(triton_kernel, shape, compile_time_config, split_k):
    grid = (
        cdiv(shape.m, compile_time_config.block_m) * cdiv(shape.n, compile_time_config.block_n),
        shape.b if shape.b > 1 else split_k,
        1,
    )
    stream_0 = get_cuda_stream_0()
    return lambda torch_inputs: triton_kernel.run(*torch_inputs, grid=grid, stream=stream_0)

def get_triton_nullary(wrapped_triton_op, torch_inputs):
    return lambda: wrapped_triton_op(torch_inputs)

CACHE_DIR = "tmp/triton_run/cache"

class TritonResultCache(PicklingResultCache):
    def __init__(self, cache_key: str):
        super().__init__(
            folder=f"{CACHE_DIR}/triton_profiler_logs", 
            cache_key=cache_key
        )

def correctness_check(triton_nullary, eager_nullary, torch_inputs, tol=1e-2):
    d = torch_inputs[-1]
    
    d.zero_()
    eager_nullary()
    d_ref = d.clone()

    d.zero_() # needed for split-k in triton since it uses atomic add
    triton_nullary()

    # if not allclose(d, d_ref, tol=tol):
    #     print(d)
    #     print(d_ref)

    assert allclose(d, d_ref, tol=tol)

class TritonBenchmark(BenchmarkBase):
    def __init__(self, parsed_args, gemm_dtype: GemmDataType, k_decomp: KDecomposition, kernel_style: TritonMMKernelStyle):
        self.kernel_style = kernel_style
        super().__init__(parsed_args, gemm_dtype, k_decomp)

        if k_decomp == KDecomposition.STREAM_K:
            raise ValueError("Triton doesn't support efficient stream-k.")
        if k_decomp in (KDecomposition.SPLIT_K_PARALLEL, KDecomposition.SPLIT_K_SERIAL_PARALLEL):
            raise ValueError("Triton doesn't support parallel split-k.")
        if k_decomp == KDecomposition.SPLIT_K_SERIAL and self.kernel_style == TritonMMKernelStyle.INDUCTOR:
            raise ValueError("Inductor-style kernels don't support split-k.")


    def read(self):
        self.shapes = CSVProblemReader.read(self.parsed_args.in_file)

        filter_fns = [
            # dtype filter
            (lambda shape: shape.data_type == self.gemm_dtype),
            # if kernel style is tritonopsmatmul, then skip batched.
            (lambda shape: not (self.kernel_style == TritonMMKernelStyle.TRITON_OPS_MATMUL and shape.op_type.is_batched())),
            # if split-k, then skip batched.
            (lambda shape: not (self.k_decomp.is_split_k() and shape.op_type.is_batched())),
            # if kernel style is tritonopsmatmul, then skip shapes with bias.
            (lambda shape: not (shape.op_type.has_bias() and self.kernel_style == TritonMMKernelStyle.TRITON_OPS_MATMUL)),
            # BUG: currently, seems the triton ops matmul code does not support bf16 atomic add
            (lambda shape: not (shape.data_type == GemmDataType.BF16 and self.kernel_style == TritonMMKernelStyle.TRITON_OPS_MATMUL and self.k_decomp.is_split_k())),
        ]

        for filter_fn in filter_fns:
            self.shapes = list(filter(filter_fn, self.shapes))

    
    def get_compile_time_configs(self):
        self.compile_time_configs = get_mm_configs(GemmFramework.BOTH, GemmFramework.TRITON, self.gemm_dtype)
        # print(len(self.compile_time_configs))
        # pprint(self.compile_time_configs)
    
    def apply_pre_build_patches(self):
        return
    
    def build(self):
        templates = {}
        for shape in self.shapes:
            for compile_time_config in self.compile_time_configs:
                split_k_max = 16 if (self.k_decomp.is_split_k() and shape.b == 1) else 1
                for split_k in range(1, split_k_max + 1):
                    template_key = (self.kernel_style, shape, compile_time_config, split_k)
                    templates[template_key] = _get_template(
                        self.kernel_style, shape, compile_time_config, self.gemm_dtype, split_k=split_k
                    )
                    # print(template_key)
                    # print(templates[template_key])
        
        triton_futures = {k: async_compile.triton(v) for (k, v) in templates.items()}
        async_compile.wait(triton_futures)
        self.triton_futures = triton_futures
        # pprint(triton_futures)
    
    def _process_profiling_result(self, shape, result, gemm_dtype, k_decomp):
        # print("--------------")
        # print(f"shape: {shape}")
        # pprint(result)
        cuda_version = os.environ["CUDA_VERSION"]
        # (kernel_style, _shape, compile_time_config, split_k)
        best_key = sorted(result.keys(), key=result.__getitem__)[0]
        kernel_style, _shape, compile_time_config, split_k = best_key
        best_runtime = result[best_key][0]

        b, m, n, k, layout = shape.b, shape.m, shape.n, shape.k, shape.layout.name

        backend = "TRITON_" + os.environ["MM_BENCH_TRITON_OPTION_ENV"].upper()
        print(",".join(map(str, [backend,cuda_version,shape.op_type.name,shape.bias_type.name,gemm_dtype.name,k_decomp.name,kernel_style.name,b,m,n,k,layout,best_runtime])))
        pass
    
    def profile_one(self, shape):
        
        result_cache = TritonResultCache(
            f"{self.cache_key}_w{self.parsed_args.warmup_iterations}_p{self.parsed_args.profiling_iterations}_t{self.parsed_args.trial_num}"
        )

        if shape in result_cache:
            results = result_cache[shape]
            self._process_profiling_result(shape, results, self.gemm_dtype, self.k_decomp)
            return
        
        torch_inputs = prep_torch_inputs(shape, self.gemm_dtype)

        results = {}

        # need to profile each of the triton_futures for the given shape
        for compile_time_config in self.compile_time_configs:
            split_k_max = 16 if (self.k_decomp.is_split_k() and shape.b == 1) else 1
            for split_k in range(1, split_k_max + 1):
                key = (self.kernel_style, shape, compile_time_config, split_k)
                triton_kernel = self.triton_futures.get(key)
                if triton_kernel is None:
                    continue
                wrapped_triton_op = get_wrapped_triton_op(triton_kernel, shape, compile_time_config, split_k)
                triton_nullary = get_triton_nullary(wrapped_triton_op, torch_inputs)

                if self.run_correctness_check:
                    eager_nullary = get_eager_nullary(shape, torch_inputs)
                    correctness_check(triton_nullary, eager_nullary, torch_inputs)

                dur = do_bench(
                    triton_nullary, 
                    warmup=self.parsed_args.warmup_iterations, 
                    rep=self.parsed_args.profiling_iterations
                )
                results[key] = dur
        
        result_cache[shape] = results
        self._process_profiling_result(shape, results, self.gemm_dtype, self.k_decomp)

    def profile(self):
        self.run_correctness_check = False
        cuda_version = os.environ["CUDA_VERSION"].replace(".", "")
        self.cache_key = hash_strings(
            [
                self.kernel_style.name, 
                self.gemm_dtype.name, 
                os.environ["MM_BENCH_TRITON_OPTION_ENV"],
                self.k_decomp.name,
                cuda_version,
            ]
        )
        for shape in self.shapes:
            self.profile_one(shape)
            sys.stdout.flush()
    
if __name__ == "__main__":

    parsed_args = parse_args(
        k_decomposition_choices = [KDecomposition.NONE, KDecomposition.SPLIT_K_SERIAL], 
        kernel_style_choices = [TritonMMKernelStyle.INDUCTOR, TritonMMKernelStyle.TRITON_OPS_MATMUL],
    )

    print("Backend,CUDA,Op Type,Bias Type,Data Type,K Decomposition,Kernel Style,B,M,N,K,Layout,Runtime")
    for gemm_data_type in GemmDataType:
        TritonBenchmark(parsed_args, gemm_data_type, parsed_args.k_decomposition, parsed_args.kernel_style).run()