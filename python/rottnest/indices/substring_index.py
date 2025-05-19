from rottnest.indices.index_interface import RottnestIndex, CacheRanges
import rottnest.rottnest as rottnest
import pyarrow
import polars
from typing import List
import numpy as np

class SubstringIndex(RottnestIndex):
    def __init__(self, char_index: bool = False, token_skip_factor: int = 1, 
                 tokenizer_file: str | None = None, 
                 token_viable_limit: int = 10,
                 index_mode: str = 'physical', brute_force_threshold: int = 1000):
        super().__init__('substring', index_mode, 'str', brute_force_threshold)
        self.char_index = char_index
        self.token_skip_factor = token_skip_factor
        self.tokenizer_file = tokenizer_file
        self.token_viable_limit = token_viable_limit
    
    def brute_force(self, data: pyarrow.Table, column_name: str, query: str, K: int) -> pyarrow.Table:
        return polars.from_arrow(data).filter(polars.col(column_name)\
            .str.to_lowercase().str.contains(query.lower(), literal=True)).head(K).to_arrow()

    def build_index(self, data_arr: pyarrow.Array, uid_arr: pyarrow.Array, index_name: str) -> CacheRanges:
        return CacheRanges(rottnest.build_lava_substring(index_name, data_arr, 
            uid_arr, self.tokenizer_file, self.token_skip_factor, self.char_index))

    def search_index(self, indices: List[str], query: str, K: int):
        return rottnest.search_lava_substring(indices, query, K, "aws",         
                sample_factor = self.token_skip_factor, token_viable_limit = self.token_viable_limit, char_index = self.char_index)
    
    def compact_indices(self, new_index_name: str, indices: List[str], offsets: np.array):
        return CacheRanges(rottnest.merge_lava_generic(f"{new_index_name}.lava", [f"{name}.lava" for name in indices], 
                                                       offsets, 3 if self.char_index else 1))
