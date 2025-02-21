#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include "eigen_ufunc.hh"

#include "pocketfft_hdronly.h"

const char* convolve_valid_doc = R"(
  Convolve waveforms with a provided kernel

  Parameters:
  -----------
  w_in : array-like
         Input Waveform
  kernel : array-like
         Convolution kernel
  w_out : array-like
          Output waveform after covolution
)";

template <typename T, int A>
void convolve_valid(const_wfblock_ref<T, A> wf_in, const_array_ref<T> kernel,  wfblock_ref<T, A> wf_out) {
  int N = std::min(wf_in.cols(), kernel.size());
  if(wf_out.cols() != std::abs(wf_in.cols() - kernel.size()) + 1)
    throw std::runtime_error("Bad dimensions for 'valid' convolution: (" +
			     std::to_string(wf_in.cols()) + "),(" +
			     std::to_string(kernel.size()) + ")->(" +
			     std::to_string(wf_out.cols()) + ")"
			     );

  wf_out = 0;
  
  for(int i_out=0; i_out<wf_out.cols(); i_out++) {
    for(int i_ker=0; i_ker<N; i_ker++) {
      wf_out.col(i_out) += wf_in.col(i_out+i_ker)*kernel[kernel.size()-1-i_ker];
    }
  }
}

add_ufunc_impl(convolve_valid_f, (convolve_valid<float, Aligned>), (convolve_valid<float, Unaligned>), "(n),(m)->(p)")
add_ufunc_impl(convolve_valid_d, (convolve_valid<double, Aligned>), (convolve_valid<double, Unaligned>), "(n),(m)->(p)")
create_ufunc(convolve_valid_ufunc,
	     "convolve_valid",
	     convolve_valid_doc,
	     convolve_valid_f,
	     convolve_valid_d
	     )


const char* convolve_full_doc = R"(
  Convolve waveforms with a provided kernel

  Parameters:
  -----------
  w_in : array-like
         Input Waveform
  kernel : array-like
         Convolution kernel
  w_out : array-like
          Output waveform after convolution
)";

template <typename T, int A>
void convolve_full(const_wfblock_ref<T, A> wf_in, const_array_ref<T> kernel,  wfblock_ref<T, A> wf_out) {
  if(wf_out.cols() != wf_in.cols() + kernel.size() - 1)
    throw std::runtime_error("Bad dimensions for convolution");

  wf_out = 0;
  
  for(int i_in=0; i_in<wf_in.cols(); i_in++) {
    for(int i_ker=0; i_ker<kernel.size(); i_ker++) {
      wf_out.col(i_in+i_ker) += wf_in.col(i_in)*kernel[kernel.size()-1-i_ker];
    }
  }
}


add_ufunc_impl(convolve_full_f, (convolve_full<float, Aligned>), (convolve_full<float, Unaligned>), "(n),(m)->(p)")
add_ufunc_impl(convolve_full_d, (convolve_full<double, Aligned>), (convolve_full<double, Unaligned>), "(n),(m)->(p)")
create_ufunc(convolve_full_ufunc,
	     "convolve_full",
	     convolve_full_doc,
	     convolve_full_f,
	     convolve_full_d
	     )


const char* convolve_doc = R"(
  Convolve waveforms with a provided kernel in the time-domain.

  Parameters:
  -----------
  w_in : array-like
         Input Waveform
  kernel : array-like
         Convolution kernel
  w_out : array-like
         Output waveform after convolution. Must be sized between
         abs(w_in.len - kernel.len) + 1 (i.e. "mode=valid") and
         w_in.len + kernel.len - 1 (i.e. "mode=same").
)";

template <typename T, int A>
void convolve(const_wfblock_ref<T, A> wf_in, const_array_ref<T> kernel,  wfblock_ref<T, A> wf_out) {
  int N1 = (wf_in.cols()+kernel.size()-wf_out.cols())/2;
  if(N1<0 || N1>kernel.size())
    throw std::runtime_error("Bad dimensions for convolution");
  wf_out = 0;
  
  for(int i_out=0; i_out<wf_out.cols(); i_out++) {
    for(int i_ker=std::max(N1-i_out, 0);
	i_ker<std::min(kernel.size(), (wf_out.cols()+N1-i_out));
	i_ker++) {
      wf_out.col(i_out) += wf_in.col(i_out-N1+i_ker)*kernel[kernel.size()-1-i_ker];
    }
  }
}

add_ufunc_impl(convolve_f, (convolve<float, Aligned>), (convolve<float, Unaligned>), "(n),(m)->(p)")
add_ufunc_impl(convolve_d, (convolve<double, Aligned>), (convolve<double, Unaligned>), "(n),(m)->(n)")
create_ufunc(convolve_ufunc,
	     "convolve",
	     convolve_doc,
	     convolve_f,
	     convolve_d
	     )


const char* convolve_same_doc = R"(
  Convolve waveforms with a provided kernel

  Parameters:
  -----------
  w_in : array-like
         Input Waveform
  kernel : array-like
         Convolution kernel
  w_out : array-like
          Output waveform after covolution
)";

template <typename T, int A>
void convolve_same(const_wfblock_ref<T, A> wf_in, const_array_ref<T> kernel,  wfblock_ref<T, A> wf_out) {
  int N1 = kernel.size()/2;
  wf_out = 0;
  
  for(int i_out=0; i_out<wf_out.cols(); i_out++) {
    for(int i_ker=std::max(N1-i_out, 0);
	i_ker<std::min(kernel.size(), (wf_out.cols()+N1-i_out));
	i_ker++) {
      wf_out.col(i_out) += wf_in.col(i_out-N1+i_ker)*kernel[kernel.size()-1-i_ker];
    }
  }
}

add_ufunc_impl(convolve_same_f, (convolve_same<float, Aligned>), (convolve_same<float, Unaligned>), "(n),(m)->(n)")
add_ufunc_impl(convolve_same_d, (convolve_same<double, Aligned>), (convolve_same<double, Unaligned>), "(n),(m)->(n)")
create_ufunc(convolve_same_ufunc,
	     "convolve_same",
	     convolve_same_doc,
	     convolve_same_f,
	     convolve_same_d
	     )



const char* fft_convolve_doc = R"(
  FFT Convolve waveforms with a provided kernel

  Parameters:
  -----------
  w_in : array-like
         Input Waveform
  kernel : array-like
         Convolution kernel
  w_out : array-like
          Output waveform after convolution
)";

template <typename T, int A>
void fft_convolve(const_wfblock_ref<T, A> wf_in, const_array_ref<T> kernel,  wfblock_ref<T, A> wf_out) {
  if(wf_out.cols() > wf_in.cols() + kernel.size() - 1 ||
     wf_out.cols() < std::abs(wf_in.cols() - kernel.size()) + 1)
    throw std::runtime_error("Bad dimensions for convolution");

  // Find smallest dim larger than size needed for FT convolution that is a multiple of 2, 3, 5, 7 and 11
  size_t ft_dim = pocketfft::detail::util::good_size_cmplx(wf_in.cols() + kernel.size() - 1);

  // Allocate memory for internal buffers and map it out
  char* buf = (char*)pocketfft::detail::aligned_alloc(Aligned, (ft_dim*sizeof(T)+Aligned)*(wf_in.rows() + kernel.rows())*2);
  size_t offset = 0;
  Eigen::Map<wfblock<T,Eigen::Dynamic>, Aligned> wf_zp((T*)buf, wf_in.rows(), ft_dim); // zero-padded time-domain waveform
  offset += (wf_zp.size()*sizeof(T)/Aligned+1)*Aligned;
  Eigen::Map<wfblock<std::complex<T>,Eigen::Dynamic>, Aligned> wf_ft((std::complex<T>*)(buf + offset), wf_in.rows(), ft_dim/2+1); // fft of waveform
  offset += (wf_ft.size()*sizeof(std::complex<T>)/Aligned+1)*Aligned;
  Eigen::Map<wfblock<T, 1>, Aligned> ker_zp((T*)(buf+offset), ft_dim); // zero-padded time-domain kernel
  offset += (ker_zp.size()*sizeof(T)/Aligned+1)*Aligned;
  Eigen::Map<wfblock<std::complex<T>, 1>, Aligned> ker_ft((std::complex<T>*)(buf+offset), ft_dim/2+1); // zero-padded time-domain kernel

  // Fill zero-padded buffers
  wf_zp.block(0, 0, wf_in.rows(), wf_in.cols()) = wf_in;
  wf_zp.block(0, wf_in.cols(), wf_in.rows(), wf_zp.cols()) = 0;
  ker_zp.segment(0, kernel.size()) = kernel;
  ker_zp.segment(kernel.size(), ker_zp.cols()) = 0;

  pocketfft::r2c(
    pocketfft::shape_t{size_t(wf_zp.rows()), size_t(wf_zp.cols())},
    pocketfft::stride_t{wf_zp.innerStride()*int(sizeof(T)), wf_zp.outerStride()*int(sizeof(T))},
    pocketfft::stride_t{wf_ft.innerStride()*int(sizeof(std::complex<T>)), wf_ft.outerStride()*int(sizeof(std::complex<T>))},
    1,
    pocketfft::FORWARD,
    reinterpret_cast<const T*>(wf_zp.data()),
    reinterpret_cast<std::complex<T>*>(wf_ft.data()),
    T(1.)
    );
  pocketfft::r2c(
    pocketfft::shape_t{size_t(ker_zp.size())},
    pocketfft::stride_t{int(sizeof(T))},
    pocketfft::stride_t{int(sizeof(std::complex<T>))},
    0,
    pocketfft::FORWARD,
    reinterpret_cast<const T*>(ker_zp.data()),
    reinterpret_cast<std::complex<T>*>(ker_ft.data()),
    T(1.)
  );
  
  wf_ft.rowwise() *= ker_ft;
  pocketfft::c2r(
    pocketfft::shape_t{size_t(wf_zp.cols()), size_t(wf_zp.rows())},
    pocketfft::stride_t{wf_ft.outerStride()*int(sizeof(std::complex<T>)), wf_ft.innerStride()*int(sizeof(std::complex<T>))},
    pocketfft::stride_t{wf_zp.outerStride()*int(sizeof(T)), wf_zp.innerStride()*int(sizeof(T))},
    0,
    pocketfft::BACKWARD,
    reinterpret_cast<const std::complex<T>*>(wf_ft.data()),
    reinterpret_cast<T*>(wf_zp.data()),
    T(1./wf_zp.cols())
  );
  wf_out = wf_zp.block(0, kernel.size()/2, wf_out.rows(), wf_out.cols());
  pocketfft::detail::aligned_dealloc(buf);
}

add_ufunc_impl(fft_convolve_f, (fft_convolve<float, Aligned>), (fft_convolve<float, Unaligned>), "(n),(m)->(p)")
add_ufunc_impl(fft_convolve_d, (fft_convolve<double, Aligned>), (fft_convolve<double, Unaligned>), "(n),(m)->(p)")
create_ufunc(fft_convolve_ufunc,
	     "fft_convolve",
	     fft_convolve_doc,
	     fft_convolve_f,
	     fft_convolve_d
	     )
