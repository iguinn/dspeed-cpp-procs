#include "eigen_ufunc.hh"

const char* derivative_doc = R"(
    Calculates mean of waveform
    Parameters
    ----------
    w_in : array-like
           waveform take mean of
    
    deriv_out : float
            mean of w_in
)";

// get mean of waveform. Mostly just implemented as a test...
template <typename T, int A>
void derivative(const_wfblock_ref<T, A> w_in, scalarblock_ref<T, A> d_out) {
  for(size_t i=0; i<d_out.cols(); i++)
    d_out.col(i) = w_in.col(i+1) - w_in.col(i);
}

add_ufunc_impl(derivative_f, (derivative<float, Aligned>), (derivative<float, Unaligned>), "(n)->(n-1)" )
add_ufunc_impl(derivative_d, (derivative<double, Aligned>), (derivative<double, Unaligned>), "(n)->(n-1)" )
create_ufunc(derivative_ufunc, "derivative", derivative_doc, derivative_f, derivative_d)
