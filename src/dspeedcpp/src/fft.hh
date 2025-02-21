#include "eigen_ufunc.hh"

#include <stdexcept>
#include <complex>
#include <map>
#include <tuple>

#include "pocketfft_hdronly.h"

const char* fft_doc = R"(
  FFT
)";


template <typename T, int A>
void fft(const_wfblock_ref<T, A> wf_in, wfblock_ref<std::complex<T>, A> dft_out) {
  if(dft_out.cols() != wf_in.cols()/2+1)
    throw std::runtime_error("Bad dimensions for r2c fft: (" + std::to_string(wf_in.cols()) + ", " + std::to_string(dft_out.cols()) + ")");

  pocketfft::shape_t shape_in{size_t(wf_in.cols()), size_t(wf_in.rows())};
  pocketfft::stride_t stride_in{
    (A>0 ? wf_in.outerStride() : wf_in.innerStride())*int(sizeof(T)),
    (A>0 ? wf_in.innerStride() : 0)*int(sizeof(T))
  };
  pocketfft::stride_t stride_out{
    (A>0 ? dft_out.outerStride() : dft_out.innerStride())*int(sizeof(std::complex<T>)),
    (A>0 ? dft_out.innerStride() : 0)*int(sizeof(std::complex<T>))
  };
  pocketfft::r2c(shape_in, stride_in, stride_out, 0, pocketfft::FORWARD, reinterpret_cast<const T*>(wf_in.data()), reinterpret_cast<std::complex<T>*>(dft_out.data()), T(1.));
}


add_ufunc_impl(fft_f, (fft<float, Aligned>), (fft<float, Unaligned>), "(n),(m)")
add_ufunc_impl(fft_d, (fft<double, Aligned>), (fft<double, Unaligned>), "(n),(m)")
create_ufunc(fft_ufunc,
	     "fft",
	     fft_doc,
	     fft_f,
	     fft_d
	     )
