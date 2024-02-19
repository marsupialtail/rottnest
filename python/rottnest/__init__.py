import rottnest as rottnest
from .pele import  search_index_natural_language, index_file_natural_language, merge_index_natural_language
from .ahupuaa import partition_periodic, partition_sessionize
from .ahupuaa import MetricsPartition as MetricsPartition

__doc__ = rottnest.__doc__
if hasattr(rottnest, "__all__"):
    __all__ = rottnest.__all__