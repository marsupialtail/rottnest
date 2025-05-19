import pyarrow
import numpy as np
from typing import List

class CacheRanges:
    def __init__(self, cache_ranges: list[tuple[int, int]]):
        self.cache_ranges = cache_ranges

class RottnestIndex:
    def __init__(self, name: str, index_mode: str = 'physical', data_type: str = 'str', brute_force_threshold: int = 1000):
        
        assert data_type in {'str', 'binary'}
        self.name = name
        self.index_mode = index_mode
        self.data_type = data_type
        self.brute_force_threshold = brute_force_threshold

    def brute_force(self, data: pyarrow.Table, column_name: str, query: str, K: int) -> pyarrow.Table:
        # Implement query logic
        raise NotImplementedError("Brute force query not implemented")

    def build_index(self, data_arr: pyarrow.Array, uid_arr: pyarrow.Array, index_name: str, **kwargs) -> CacheRanges:
        """
        Implement build index logic. This will be called by Rottnest to index new files.
        Returns a dictionary of cache ranges.
        Args:
            data_arr: pyarrow Array of the data to be indexed.
        
        """
        raise NotImplementedError("Index build logic ")

    def search_index(self, indices: List[str], query: str, K: int) -> List[tuple[int, int]]:
        raise NotImplementedError("Search index logic not implemented")

    def compact_indices(self, new_file_name: str, input_file_names: List[str], offsets: np.array):
        # Implement compact indices logic
        raise NotImplementedError("Compact indices logic not implemented")

