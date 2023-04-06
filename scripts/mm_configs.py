from dataclasses import dataclass, astuple
import functools
from typing import List, Tuple

from common_types import GemmDataType, GemmFramework

@dataclass(frozen=True)
class MatmulConfig:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int

    def __lt__(self, other):
        return astuple(self) < astuple(other)


def next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n"""
    assert n <= 2**32, "32-bit only"
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


# We currently do not support torchinductor's method of dynamically selecting configs
# with reduced threadblock shapes when given smaller shapes. In principle, this is 
# not too difficult to implement, but found that CUTLASS is not able to support some
# of the smaller block sizes. As a result, we currently opt to remove support for that
# feature for easier apples-to-apples comparisons.
def unfiltered_configs(
    configs: List[Tuple[int, int, int, int, int]]
):
    """Heuristic to shrink configs when they are bigger than the input size"""
    used = set()
    for block_m, block_n, block_k, num_stages, num_warps in configs:
        if (block_m, block_n, block_k, num_stages, num_warps) not in used:
            used.add((block_m, block_n, block_k, num_stages, num_warps))
            yield MatmulConfig(
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                num_stages=num_stages,
                num_warps=num_warps,
            )

triton_mm_configs = [
    # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
    (64, 64, 32, 2, 4),
    (64, 128, 32, 3, 4),
    (128, 64, 32, 3, 4),
    (64, 128, 32, 4, 8),
    (128, 64, 32, 4, 8),
    (64, 32, 32, 5, 8),
    (32, 64, 32, 5, 8),
    (128, 128, 32, 2, 8),
    (64, 64, 64, 3, 8),
    (32, 32, 128, 2, 4),
    (64, 64, 16, 2, 4),
    (32, 32, 16, 1, 2),
]

cutlass_mm_configs_f16_bf16 = [
    (256, 128, 32, 3, 8),
    (128, 256, 32, 3, 8),
    (256, 64, 32, 3, 4),
    (256, 64, 32, 4, 4),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 3, 4),
    (128, 128, 32, 4, 4),
    (128, 128, 32, 5, 4),
    (128, 64, 32, 6, 4),
    (64, 128, 32, 6, 4),
    (64, 64, 32, 10, 4),
    (256, 128, 64, 3, 8),
    (128, 256, 64, 3, 8),
    (256, 64, 64, 4, 4),
    (64, 256, 64, 4, 4),
    (128, 128, 64, 4, 4),
    (256, 64, 64, 3, 4),
    (64, 256, 64, 3, 4),
    (128, 128, 64, 3, 4),
    (128, 64, 64, 3, 4),
    (64, 128, 64, 3, 4),
    (64, 64, 64, 5, 4),
]

cutlass_mm_configs_tf32 = [
    (256, 128, 16, 3, 8),
    (128, 256, 16, 3, 8),
    (256, 64, 16, 4, 4),
    (64, 256, 16, 4, 4),
    (128, 128, 16, 5, 4),
    (128, 128, 16, 4, 4),
    (128, 128, 16, 3, 4),
    (128, 64, 16, 6, 4),
    (64, 128, 16, 6, 4),
    (64, 64, 16, 10, 4),
    (256, 128, 32, 3, 8),
    (128, 256, 32, 3, 8),
    (256, 64, 32, 4, 4),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 128, 32, 3, 4),
    (128, 64, 32, 3, 4),
    (64, 128, 32, 3, 4),
    (64, 64, 32, 5, 4),
]

def get_mm_configs(config_source: GemmFramework, config_consumer: GemmFramework, dtype: GemmDataType):
    configs = []

    if config_source in (GemmFramework.CUTLASS, GemmFramework.BOTH):
        if dtype in (GemmDataType.F16, GemmDataType.BF16):
            configs += list(cutlass_mm_configs_f16_bf16)
        elif dtype == GemmDataType.TF32:
            configs += list(cutlass_mm_configs_tf32)
        else:
            raise NotImplementedError

    if config_source in (GemmFramework.TRITON, GemmFramework.BOTH):
        configs += triton_mm_configs

    configs = sorted(set(configs))

    # print(f"Before pruning, {len(configs)} configs are present.")
    if config_consumer in (GemmFramework.CUTLASS, GemmFramework.BOTH):
        if dtype == GemmDataType.TF32:
            configs = sorted(filter(lambda x: x[2] <= 32, configs))
        elif dtype in (GemmDataType.F16, GemmDataType.BF16):
            configs = sorted(filter(lambda x: x[0] >= 64 and x[1] >= 64 and x[2] >= 32, configs))
    # print(f"After pruning, {len(configs)} configs are present.")

    configs = [MatmulConfig(*config_as_tuple) for config_as_tuple in configs]

    return configs

def _argsort(lst, **kwargs):
    return sorted(range(len(lst)),key=lst.__getitem__, **kwargs)

def get_cutlass_warp_counts(matmul_config: MatmulConfig) -> Tuple[int, int, int]:
    block_mn = [matmul_config.block_m, matmul_config.block_n]
    num_warps = matmul_config.num_warps
    curr_num_warps = num_warps
    warp_counts = [1, 1, 1]
    idx = 0
    MIN_BLOCK_SIZE = 16
    while curr_num_warps > 1:
        if block_mn[0] != block_mn[1]:
            idx = _argsort(block_mn, reverse=True)[0]
        if block_mn[idx] > MIN_BLOCK_SIZE:
            block_mn[idx] /= 2
            warp_counts[idx] *= 2
            curr_num_warps /= 2
        idx = 1 - idx
    return tuple(warp_counts)