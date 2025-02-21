# dspeed-cpp-procs
High performance processors for dspeed

This package includes processors for use in dspeed that are optimized using C++ in order to take advantage of hardware level SIMD commands, and wrapped in numpy's ufunc interface.

To install:
```
git clone --recursive --shallow-submodules https://github.com/iguinn/dspeed-cpp-procs
cd dspeed-cpp-procs
pip install [--user] .
```

To get the most out of these processors, we want the memory to be memory aligned and vectorized in blocks of size 16. The `aligned_ndarray` module will generate numpy arrays with these properties.
```
import dspeedcpp
dspeedcpp.aligned_ndarray.zeros( (16, 1000) )
dspeed.cpp.processors.mean(a)
```

The processors currently implemented are: convolve, convolve_full, convolve_same, convolve_valid, fft, fft_convolve, mean, pole_zero, trap_filter, trap_norm
