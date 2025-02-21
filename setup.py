 #!/usr/bin/env python3
from setuptools import setup, Extension
import numpy

# For list of supported architectures see: https://eigen.tuxfamily.org/index.php?title=FAQ#Which_SIMD_instruction_sets_are_supported_by_Eigen.3F
# I've commented out some that don't seem to make a difference
simd_args = {
    "sse2": ["-msse2", "-DSIMD_ALIGN=Eigen::Aligned16"],
    "sse3": ["-msse3", "-DSIMD_ALIGN=Eigen::Aligned16"],
    "sse4": ["-msse4.2"],
    "avx": ["-mavx", "-DSIMD_ALIGN=Eigen::Aligned64"],
    "avx2": ["-mavx2", "-mfma", "-DSIMD_ALIGN=Eigen::Aligned64"],
    "avx512f": ["-mavx512f", "-mfma", "-mavx2", "-DSIMD_ALIGN=Eigen::Aligned64"],
}

setup_args = {
    "ext_modules": [ Extension(
        name = f"dspeedcpp.processors_{suffix}",
        sources = [
            "src/dspeedcpp/src/dspeedcpp_module.cpp",
        ],
        include_dirs = [
            "src/dspeedcpp/src",
            "include/eigen",
            "include/pocketfft",
            numpy.get_include(),
        ],
        depends = [
            "srs/dspeedcpp/src/eigen_ufunc.hh",
            "src/dspeedcpp/src/mean.hh",
            "src/dspeedcpp/src/pole_zero.hh",
            "src/dspeedcpp/src/trap_filter.hh",
            "src/dspeedcpp/src/trap_norm.hh",
            "src/dspeedcpp/src/convolve.hh",
            "src/dspeedcpp/src/fft.hh",
        ],
        define_macros = [
            ("module_name", f"processors_{suffix}"),
            ("POCKETFFT_NO_MULTITHREADING", "1"),
            ("POCKETFFT_CACHE_SIZE", "100"),
        ],
        extra_compile_args=["-std=c++20", *arg],
        language="c++",
        py_limited_api=True
    ) for suffix, arg in simd_args.items() ]
}
setup(**setup_args)
