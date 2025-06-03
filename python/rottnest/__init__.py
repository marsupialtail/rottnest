import rottnest as rottnest
from .backends.utils import  get_fs_from_file_path, get_physical_layout, group_mergeable_indices, index_files, merge_indices, search_index, search_parquet_lake
from .backends.iceberg import IcebergBackend
import rottnest.backends.s3_utils as s3_utils
from .indices.index_interface import RottnestIndex
from .indices.bm25_index import Bm25Index
from .indices.substring_index import SubstringIndex
from .indices.uuid_index import UuidIndex
from .indices.vector_index import VectorIndex

__doc__ = rottnest.__doc__
if hasattr(rottnest, "__all__"):
    __all__ = rottnest.__all__