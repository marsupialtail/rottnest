from rottnest.indices.index_interface import RottnestIndex, CacheRanges
import rottnest.rottnest as rottnest
import pyarrow
import polars
from typing import List
import pyarrow.compute as pac
import numpy as np

class UuidIndex(RottnestIndex):
    def __init__(self, index_mode: str = 'physical', brute_force_threshold: int = 1000):
        super().__init__('uuid', index_mode, 'str', brute_force_threshold)
    
    def brute_force(self, data: pyarrow.Table, column_name: str, query: str, K: int) -> pyarrow.Table:
        return polars.from_arrow(data).filter(polars.col(column_name)\
            .str.to_lowercase().str.contains(query.lower(), literal=True)).head(K).to_arrow()

    def build_index(self, data_arr: pyarrow.Array, uid_arr: pyarrow.Array, index_name: str) -> CacheRanges:
        idx = pac.sort_indices(data_arr)
        data_arr = data_arr.take(idx)
        uid_arr = uid_arr.take(idx)
        return CacheRanges(rottnest.build_lava_uuid(index_name, data_arr, uid_arr))

    def search_index(self, indices: List[str], query: str, K: int):
        return rottnest.search_lava_uuid(indices, query, K, "aws")

    def compact_indices(self, new_index_name: str, indices: List[str], offsets: np.array):
        return CacheRanges(rottnest.merge_lava_generic(f"{new_index_name}.lava", [f"{name}.lava" for name in indices], 
                                                       offsets, 2))