import pyarrow
from .internal import index_files_bm25, index_files_substring, index_files_vector, index_files_uuid

class RottnestIndex:
    def __init__(self, index_mode: str = 'physical'):
        self.index_mode = index_mode

    def brute_force(self, data: pyarrow.Table) -> pyarrow.Table:
        # Implement query logic
        raise NotImplementedError("Brute force query not implemented")

    def build_index(self, data_arr: pyarrow.Array, uid_arr: pyarrow.Array, output_file_name: str, **kwargs) -> dict:
        """
        Implement build index logic. This will be called by Rottnest to index new files.
        Returns a dictionary of cache ranges.
        Args:
            data_arr: pyarrow Array of the data to be indexed.
        
        """
        raise NotImplementedError("Index build logic ")

    def compact_indices(self, new_file_name: str, input_file_names: str, offsets: np.array | list):
        # Implement compact indices logic
        pass