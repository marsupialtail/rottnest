import rottnest as rottnest
from . import internal
from . import iceberg

__doc__ = rottnest.__doc__
if hasattr(rottnest, "__all__"):
    __all__ = rottnest.__all__