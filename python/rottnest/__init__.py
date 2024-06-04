import rottnest as rottnest
from .pele import  search_index_bm25, search_index_substring, search_index_vector, search_index_uuid,\
    merge_index_bm25, merge_index_substring, merge_index_vector, \
    index_file_bm25, index_file_substring, index_file_vector, index_file_kmer, search_index_vector_mem, \
    index_file_uuid

__doc__ = rottnest.__doc__
if hasattr(rottnest, "__all__"):
    __all__ = rottnest.__all__