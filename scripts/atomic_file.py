import os
import pickle
import tempfile
from typing import Union

from io_utils import make_directories

class AtomicFile:
    def __init__(self, fname: str, bin=False):
        self.fname = fname
        self.bin = bin
    
    def read(self) -> Union[bytes, str]:
        with open(self.fname, "rb" if self.bin else "r") as f:
            return f.read()
    
    def write(self, text: Union[bytes, str]):
        with tempfile.NamedTemporaryFile(mode='wb' if self.bin else "w", delete=False) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        
        os.replace(f.name, self.fname)
    
    def nonatomic_append(self, text: Union[bytes, str]):
        self.write(self.read() + text)

class ResultCache:
    def __init__(self, folder: str, cache_key: str):
        self.folder = folder
        self.cache_key = cache_key
    
    def __getitem__(self, shape):
        if shape not in self:
            raise KeyError()
        return AtomicFile(self._get_fname(shape)).read()

    def __setitem__(self, shape, value):
        fname = self._get_fname(shape)
        make_directories(fname)
        AtomicFile(fname).write(value)
    
    def _get_fname(self, shape):
        shape_key = f"{shape.op_type.name}_{shape.bias_type.name}_{shape.data_type.name}_{shape.b}_{shape.m}_{shape.n}_{shape.k}_{shape.layout.name}"
        fname = f"{self.folder}/{self.cache_key}/{shape_key}.txt"
        return fname

    def __contains__(self, shape):
        return os.path.exists(self._get_fname(shape))

class PicklingResultCache(ResultCache):
    def __init__(self, folder: str, cache_key: str):
        super().__init__(folder, cache_key)
    
    def __setitem__(self, shape, value):
        pickled_str = pickle.dumps(value, 0).decode()
        super().__setitem__(shape, pickled_str)

    def __getitem__(self, shape):
        pickled_str = super().__getitem__(shape)
        pickled_bytes = pickled_str.encode()
        value = pickle.loads(pickled_bytes)
        return value