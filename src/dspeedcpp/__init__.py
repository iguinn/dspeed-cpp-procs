from numpy.core._multiarray_umath import __cpu_features__

from .aligned_ndarray import *
from . import processors_sse3, processors_sse4, processors_avx, processors_avx2

if __cpu_features__["AVX2"]:
    from .processors_avx2 import *
elif __cpu_features__["SSE42"]:
    from .processors_sse4 import *
