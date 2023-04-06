from enum import Enum

class BiasType(Enum):
    # NOTE: bias type layout is always row major.
    NONE = 1

    # SCALAR = 2          # (1)        -- Currently not supported.
    VECTOR = 3            # (n)
    MATRIX = 4            # (m, n)
    # BATCHED_SCALAR = 5  # (b, 1, 1)  -- Currently not supported.
    BATCHED_VECTOR = 6    # (b, 1, n)
    BATCHED_MATRIX = 7    # (b, m, n)

    def is_none(self):
        return self is BiasType.NONE

    def is_batched(self):
        return self in (BiasType.BATCHED_VECTOR, BiasType.BATCHED_MATRIX)

    def is_vector(self):
        return self in (BiasType.VECTOR, BiasType.BATCHED_VECTOR)

class GemmOpType(Enum):
    MM = 1
    ADDMM = 2
    BMM = 3
    BADDBMM = 4

    def is_batched(self):
        return self in (GemmOpType.BMM, GemmOpType.BADDBMM)

    def has_bias(self):
        return self in (GemmOpType.ADDMM, GemmOpType.BADDBMM)

class GemmDataType(Enum):
    BF16 = 1
    F16 = 2
    TF32 = 3

class GemmFramework(Enum):
    CUTLASS = 1
    TRITON = 2
    BOTH = 3

class KDecomposition(Enum):
    NONE = 1
    SPLIT_K_SERIAL = 2
    SPLIT_K_PARALLEL = 3
    SPLIT_K_SERIAL_PARALLEL = 4
    STREAM_K = 5

    def is_split_k(self):
        return self in (
            KDecomposition.SPLIT_K_SERIAL,
            KDecomposition.SPLIT_K_PARALLEL,
            KDecomposition.SPLIT_K_SERIAL_PARALLEL,
        )

    def split_k_shortname(self):
        if self == KDecomposition.SPLIT_K_SERIAL:
            return "serial"
        elif self == KDecomposition.SPLIT_K_PARALLEL:
            return "parallel"
        elif self == KDecomposition.SPLIT_K_SERIAL_PARALLEL:
            return "serial,parallel"
        else:
            raise ValueError

# There are two variants in the way the matmul kernels are written and commonly used today.
# The first is the way it's currently written in inductor. The other is what the trunk
# triton.ops.matmul code looks like. They mainly differ in the block-k loop ordering and
# increments. We want to expose both approaches so they can both be tested.
class TritonMMKernelStyle(Enum):
    NONE = 1
    INDUCTOR = 2
    TRITON_OPS_MATMUL = 3