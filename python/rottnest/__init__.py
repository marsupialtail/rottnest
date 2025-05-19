import rottnest as rottnest
from .backends import iceberg
from .indices.index_interface import RottnestIndex
from .indices.bm25_index import Bm25Index
from .indices.substring_index import SubstringIndex
from .indices.uuid_index import UuidIndex
from .indices.vector_index import VectorIndex

__doc__ = rottnest.__doc__
if hasattr(rottnest, "__all__"):
    __all__ = rottnest.__all__