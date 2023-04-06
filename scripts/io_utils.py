from dataclasses import dataclass
from enum import Enum
import os
import hashlib
import shutil
from common_types import GemmOpType, BiasType, GemmDataType

class Layout(Enum):
    """
    Example: RC means row-major A, col-major B, row-major out.
    out is always row-major, which matches the standard
    PyTorch output.
    """
    RR = 1
    RC = 2
    CR = 3
    CC = 4


@dataclass(frozen=True)
class ProblemSpec:
    op_type: GemmOpType
    bias_type: BiasType
    data_type: GemmDataType
    b: int
    m: int
    n: int
    k: int
    layout: Layout

    def __post_init__(self):
        if self.op_type in (GemmOpType.MM, GemmOpType.ADDMM) and self.b != 1:
            raise ValueError(f"{self} has op_type {self.op_type}, but has batch size {self.b}. Batch size must be 1.")
        if not self.op_type.is_batched() and self.bias_type.is_batched():
            raise ValueError(f"op_type ({self.op_type}) is not batched, but bias_type ({self.bias_type}) is batched.")
        if (not self.op_type.has_bias()) and (self.bias_type is not BiasType.NONE):
            raise ValueError(f"op_type {self.op_type} does not have bias, but was supplied with non-NONE bias_type ({self.bias_type}).")
        if self.op_type.has_bias() and (self.bias_type is BiasType.NONE):
            raise ValueError(f"op_type {self.op_type} does has bias, but was supplied with NONE bias_type ({self.bias_type}).")


class TestReader:
    @staticmethod
    def read(fname):
        dtype = GemmDataType.BF16
        return [
            # ProblemSpec(1, 8, 8, 8, Layout.RC),
            ProblemSpec(GemmOpType.MM, BiasType.NONE, dtype, 1, 1200, 80, 600, Layout.RC),
            ProblemSpec(GemmOpType.MM, BiasType.NONE, dtype, 1, 1200, 80, 600, Layout.RR),
            ProblemSpec(GemmOpType.ADDMM, BiasType.VECTOR, dtype, 1, 360, 400, 8, Layout.RR),
            ProblemSpec(GemmOpType.ADDMM, BiasType.VECTOR, dtype, 1, 360, 400, 8, Layout.CC),
            ProblemSpec(GemmOpType.ADDMM, BiasType.MATRIX, dtype, 1, 360, 400, 8, Layout.RR),
            ProblemSpec(GemmOpType.ADDMM, BiasType.MATRIX, dtype, 1, 360, 400, 8, Layout.CR),
            
            # baseline
            # ProblemSpec(GemmOpType.BMM, BiasType.NONE, dtype, 100, 3600, 4000, 8, Layout.RR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.VECTOR, dtype, 100, 3600, 4000, 8, Layout.RR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.BATCHED_VECTOR, dtype, 100, 3600, 4000, 8, Layout.RR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.MATRIX, dtype, 100, 3600, 4000, 8, Layout.RR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.BATCHED_MATRIX, dtype, 100, 3600, 4000, 8, Layout.RR),

            # ProblemSpec(GemmOpType.BMM, BiasType.NONE, dtype, 100, 3600, 4000, 8, Layout.CR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.VECTOR, dtype, 100, 3600, 4000, 8, Layout.CR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.BATCHED_VECTOR, dtype, 100, 3600, 4000, 8, Layout.CR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.MATRIX, dtype, 100, 3600, 4000, 8, Layout.CR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.BATCHED_MATRIX, dtype, 100, 3600, 4000, 8, Layout.CR),

            # try
            # ProblemSpec(GemmOpType.BMM, BiasType.NONE, dtype, 184, 184, 184, 184, Layout.RR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.VECTOR, dtype, 184, 184, 184, 184, Layout.RR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.BATCHED_VECTOR, dtype, 184, 184, 184, 184, Layout.RR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.MATRIX, dtype, 184, 184, 184, 184, Layout.RR),
            # ProblemSpec(GemmOpType.BADDBMM, BiasType.BATCHED_MATRIX, dtype, 184, 184, 184, 184, Layout.RR),


            # ProblemSpec(1, 1202, 82, 602, Layout.CC),
            # ProblemSpec(1, 1201, 81, 601, Layout.RR),
            # ProblemSpec(2, 100, 80, 60, Layout.CR),
            # ProblemSpec(GemmOpType.BMM, 32, 1000, 800, 600, Layout.CR),
        ]

class CSVProblemReader:
    @staticmethod
    def read(fname):
        shapes = []
        with open(fname, "r") as f:
            for line in f.readlines():
                if line.strip() == "":
                    continue
                
                row = [entry.strip() for entry in line.split(",")]
                gemm_op_type_str, bias_type_str, data_type_str, b, m, n, k, layout_str = row
                shape = ProblemSpec(
                    GemmOpType[gemm_op_type_str], 
                    BiasType[bias_type_str], 
                    GemmDataType[data_type_str], 
                    int(b), 
                    int(m), 
                    int(n),
                    int(k), 
                    Layout[layout_str]
                )
                shapes.append(shape)
        return shapes


def get_file_hash(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()[:20]

def make_directories(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

def hash_directory_contents(directory):
    sha256_hash = hashlib.sha256()
    for dirpath, _dirnames, filenames in sorted(os.walk(directory)):
        for filename in sorted(filenames):
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "rb") as f:
                sha256_hash.update(filename.encode("utf-8"))
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
    return sha256_hash.hexdigest()[:20]

def remove_directory_if_exists(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

def hash_strings(string_list):
    sha256_hash = hashlib.sha256()
    for string in string_list:
        sha256_hash.update(string.encode('utf-8'))
    return sha256_hash.hexdigest()[:20]

def replace_in_file(fname, oldvalue, newvalue):
    with open(fname, "r") as f:
        text = f.read()
    
    modified_text = text.replace(oldvalue, newvalue)
    
    with open(fname, "w") as f:
        f.write(modified_text)