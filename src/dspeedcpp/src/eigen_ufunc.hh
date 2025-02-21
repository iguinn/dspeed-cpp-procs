/* This header contains a variety of classes, functions, and macros that
 * are helpful for building a numpy-compatible ufunc from code that uses
 * the eigen template library.
 */
#ifndef _EIGEN_UFUNC_HH_
#define _EIGEN_UFUNC_HH_

#include <tuple>
#include <vector>
#include <map>
#include <regex>
#include <type_traits>
#include <stdexcept>
#include <iostream>

#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#include "Eigen/Core"


// The first part of this contains all of the typedefs that we would need
// to work with eigen. These could be placed in a header file.
#ifndef SIMD_ALIGN
const int Aligned = Eigen::Aligned64;
#else
const int Aligned = SIMD_ALIGN;
#endif
const int Unaligned = Eigen::Unaligned;

// Data storage classes. Do not pass these to functions, to avoid copying. If
// used for tmp variable storage, use static thread_local to avoid reallocating
// on each call
template<typename T, int blocksize>
using wfblock     = Eigen::Array<T, blocksize, Eigen::Dynamic>;

template<typename T, int blocksize>
using scalarblock = Eigen::Array<T, blocksize, 1>;

// Writable references to data holding classes. Use instead of C references to
// data holders, so that you are compatible with any data configuration!
template<typename T, int Align>
using wfblock_ref    = Eigen::Ref<wfblock<T, (Align>0 ? Align/sizeof(T) : 1)>, Align, typename std::conditional< (Align>0), Eigen::OuterStride<Eigen::Dynamic>, Eigen::InnerStride<Eigen::Dynamic> >::type >;

template<typename T, int Align>
using scalarblock_ref = Eigen::Ref<scalarblock<T, (Align>0 ? Align/sizeof(T) : 1)>, Align, typename std::conditional< (Align>0), Eigen::OuterStride<Eigen::Dynamic>, Eigen::InnerStride<Eigen::Dynamic> >::type >;

// Read only references to data holding class or array expressions. Use this
// for inputs, so that you can send in things like replications of scalars!
template<typename T, int Align>
using const_wfblock_ref    = Eigen::Ref<const wfblock<T, (Align>0 ? Align/sizeof(T) : 1)>, Align, typename std::conditional< (Align>0), Eigen::OuterStride<Eigen::Dynamic>, Eigen::InnerStride<Eigen::Dynamic> >::type >;

template<typename T, int Align>
using const_scalarblock_ref = Eigen::Ref<const scalarblock<T, (Align>0 ? Align/sizeof(T) : 1)>, Align, Eigen::OuterStride<1> >;

template<typename T>
using const_array_ref = Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, 1>, Unaligned, Eigen::InnerStride<> >;

template<typename T>
constexpr char get_typechar();
template<> constexpr char get_typechar<bool>() { return NPY_BOOL; }
template<> constexpr char get_typechar<int8_t>() { return NPY_BYTE; } 
template<> constexpr char get_typechar<int16_t>() { return NPY_SHORT; } 
template<> constexpr char get_typechar<int32_t>() { return NPY_INT; } 
template<> constexpr char get_typechar<int64_t>() { return NPY_LONG; } 
template<> constexpr char get_typechar<uint8_t>() { return NPY_UBYTE; } 
template<> constexpr char get_typechar<uint16_t>() { return NPY_USHORT; } 
template<> constexpr char get_typechar<uint32_t>() { return NPY_UINT; } 
template<> constexpr char get_typechar<uint64_t>() { return NPY_ULONG; } 
template<> constexpr char get_typechar<float>() { return NPY_FLOAT; } 
template<> constexpr char get_typechar<double>() { return NPY_DOUBLE; } 
template<> constexpr char get_typechar<std::complex<float> >() { return NPY_CFLOAT; } 
template<> constexpr char get_typechar<std::complex<double> >() { return NPY_CDOUBLE; } 


// Helper with information and methods useful for handling each object we
// are allowing as arguments (i.e. arithmetic types and the typedefs above)
// Use to check compatibility and convert from numpy arrays to eigen arrays
template<typename T>
struct arg_info {
  using dtype = T;
  static const char dtype_char = get_typechar<dtype>();
  using argtype = T;
  using basetype = T;
  static const bool has_inner_dim = false;
  static const bool is_const = true;
  static const int blocksize = 0;
  using stride = Eigen::Stride<1, 1>;
  
  static bool is_aligned(char* args, npy_intp outer_dim, npy_intp outer_step)
  { return outer_step==0; }
  
  static argtype get_arg(char* args, int i_outer, int n_wfs, int inner_dim, int outer_step, int inner_step) {
    return *(argtype*)(args + i_outer*outer_step);
  }
};

template<typename T, int Align>
struct arg_info<wfblock_ref<T, Align> > {
  using argtype = wfblock_ref<T, Align>;
  using dtype = T;
  static const int blocksize = argtype::RowsAtCompileTime;
  using basetype = wfblock<dtype, blocksize>;
  static const char dtype_char = get_typechar<dtype>();
  static const bool has_inner_dim = true;
  static const bool is_const = false;
  using stride = typename std::conditional< (Align>0), Eigen::OuterStride<Eigen::Dynamic>, Eigen::InnerStride<Eigen::Dynamic> >::type;
  
  static bool is_aligned(char* args, npy_intp outer_dim, npy_intp outer_step) {
    return reinterpret_cast<std::uintptr_t>(args)%Align == 0 &&
      outer_dim%blocksize == 0 && outer_step == sizeof(T);
  }
  
  static argtype get_arg(char* args, int i_outer, int n_wfs, int inner_dim, int outer_step, int inner_step) {
    return Eigen::Map< basetype, Align, stride>( (dtype*)(args + i_outer*outer_step), n_wfs, inner_dim, stride(inner_step/sizeof(dtype)/*, outer_step/sizeof(dtype)*/) );
  }
};

template<typename T, int Align>
struct arg_info<const_wfblock_ref<T, Align> > {
  using argtype = const_wfblock_ref<T, Align>;
  using dtype = T;
  static const int blocksize = argtype::RowsAtCompileTime;
  using basetype = wfblock<dtype, blocksize>;
  static const char dtype_char = get_typechar<dtype>();
  static const bool has_inner_dim = true;
  static const bool is_const = true;
  using stride = typename std::conditional< (Align>0), Eigen::OuterStride<Eigen::Dynamic>, Eigen::InnerStride<Eigen::Dynamic> >::type;
  
  static bool is_aligned(char* args, npy_intp outer_dim, npy_intp outer_step) {
    return reinterpret_cast<std::uintptr_t>(args)%Align == 0 &&
      outer_dim%blocksize == 0 && outer_step == sizeof(T);
  }
  
  static argtype get_arg(char* args, int i_outer, int n_wfs, int inner_dim, int outer_step, int inner_step) {
    return Eigen::Map< const basetype, Align, stride>( (dtype*)(args + i_outer*outer_step), n_wfs, inner_dim, stride(inner_step/sizeof(dtype)) );
  }
};

template<typename T, int Align>
struct arg_info<scalarblock_ref<T, Align> > {
  using argtype = scalarblock_ref<T, Align>;
  using dtype = T;
  static const int blocksize = argtype::RowsAtCompileTime;
  using basetype = scalarblock<dtype, blocksize>;
  static const char dtype_char = get_typechar<dtype>();
  static const bool has_inner_dim = false;
  static const bool is_const = false;
  using stride = Eigen::InnerStride<(Align > 0 ? 1 : Eigen::Dynamic)>;
  
  static bool is_aligned(char* args, npy_intp outer_dim, npy_intp outer_step) {
    return reinterpret_cast<std::uintptr_t>(args)%Align == 0 &&
      outer_dim%blocksize == 0 && outer_step == sizeof(dtype);
  }
  
  static argtype get_arg(char* args, int i_outer, int n_wfs, int inner_dim, int outer_step, int inner_step) {
    return Eigen::Map< basetype, Align, stride>( (dtype*)(args + i_outer*outer_step), inner_dim, n_wfs, stride(outer_step/sizeof(dtype)) );
  }
};

template<typename T, int Align>
struct arg_info<const_scalarblock_ref<T, Align> > {
  using argtype = const_scalarblock_ref<T, Align>;
  using dtype = T;
  static const int blocksize = argtype::RowsAtCompileTime;
  using basetype = scalarblock<dtype, blocksize>;
  static const char dtype_char = get_typechar<dtype>();
  static const bool has_inner_dim = false;
  static const bool is_const = true;
  using stride = Eigen::InnerStride<(Align > 0 ? 1 : Eigen::Dynamic)>;
  
  static bool is_aligned(char* args, npy_intp outer_dim, npy_intp outer_step) {
    return (reinterpret_cast<std::uintptr_t>(args)%Align == 0 &&
      outer_dim%blocksize == 0 && outer_step == sizeof(dtype))
      || outer_step==0;
  }
  
  static basetype get_arg(char* args, int i_wf, int n_wfs, int inner_dim, int outer_step, int inner_step) {
    if(outer_step==0) {
      return basetype::Constant(*(dtype*)(args));
    } else {
      return Eigen::Map<const basetype, Align, stride>( (dtype*)(args + i_wf*outer_step), inner_dim, n_wfs, stride(outer_step/sizeof(dtype)) );
    }
  }
};

template<typename T>
struct arg_info<const_array_ref<T> > {
  using argtype = const_array_ref<T>;
  using dtype = T;
  static const int blocksize = 0;
  using basetype = Eigen::Array<dtype, Eigen::Dynamic, 1>;
  static const char dtype_char = get_typechar<dtype>();
  static const bool has_inner_dim = true;
  static const bool is_const = true;
  using stride = Eigen::InnerStride<>;
  
  static bool is_aligned(char* args, npy_intp outer_dim, npy_intp outer_step) {
    return outer_step == 0;
  }
  
  static argtype get_arg(char* args, int i_outer, int n_wfs, int inner_dim, int outer_step, int inner_step) {
    return Eigen::Map< const basetype, Unaligned, stride>( (dtype*)(args + i_outer*outer_step), inner_dim, stride(inner_step/sizeof(dtype)) );
  }
};



// Convert python ufunc values to tuple of eigen classes used by our function
template<typename T, typename... T_others>
auto get_args(char** args, const npy_intp* dims, const size_t* i_innerdim, const npy_intp* steps, int i_wf, int n_wfs, size_t i_outerstep=0, size_t i_innerstep=sizeof...(T_others)+1) {
  auto&& arg = std::make_tuple(arg_info<T>::get_arg(args[i_outerstep], i_wf, n_wfs, dims[*i_innerdim], steps[i_outerstep], steps[i_innerstep]));
  if(arg_info<T>::has_inner_dim) i_innerstep++;
  i_outerstep++;
  i_innerdim++;
  
  if constexpr (sizeof...(T_others)==0 )
    return arg;
  else
    return std::tuple_cat(arg, get_args<T_others...>(args, dims, i_innerdim, steps, i_wf, n_wfs, i_outerstep, i_innerstep) ); 
}



// Convert the ufunc inputs into arguments for our function and execute it
template<typename... T_align, typename... T_unalign>
void execute_ufunc( void(*f_align)(T_align...), void(*f_unalign)(T_unalign...), const size_t* i_innerdim, char** args, const npy_intp* dims, const npy_intp* steps) {

  static_assert( sizeof...(T_align)==sizeof...(T_unalign) &&
		 ( (arg_info<T_align>::dtype_char==arg_info<T_unalign>::dtype_char &&
		    arg_info<T_align>::has_inner_dim==arg_info<T_unalign>::has_inner_dim &&
		    arg_info<T_align>::is_const==arg_info<T_unalign>::is_const) && ...),
		 "Both functions must have compatible signatures!");

  
  // Get the expected blocksize
  int blocksize = std::max({arg_info<T_align>::blocksize...});
  
  // Fold expression to check if each blocksize is the same or 0, and if everything is memory-aligned
  char** it_args = args;
  const npy_intp* it_steps = steps;
  bool aligned = ( ( (arg_info<T_align>::blocksize==0 || arg_info<T_align>::blocksize==blocksize) && arg_info<T_align>::is_aligned(*(it_args++), dims[0], *(it_steps++)) ) && ...);
  
  if(aligned) {
    // Process blocks of waveforms using the aligned function
    for(int i_wf=0; i_wf<dims[0]; i_wf += blocksize)
      std::apply((f_align), get_args<T_align...>(args, dims, i_innerdim, steps, i_wf, blocksize));
  } else {
    // Process each WF 1 at a time using the un-aligned function
    for(int i_wf=0; i_wf<dims[0]; i_wf++)
      std::apply((f_unalign), get_args<T_unalign...>(args, dims, i_innerdim, steps, i_wf, 1));
  }
}


// Get number of arguments of function
template<typename... T>
constexpr size_t get_nargs(void(*f)(T...))
{ return sizeof...(T); }

// parse signature into map of inner dimensions and number of input args
constexpr const size_t* const parse_sig(std::string_view signature) {
  size_t* i_dim = new size_t[signature.size()];
  char dim_list[signature.size()];
  size_t n_dims = 0;
  
  for(size_t i = 0; i<signature.size(); i++) {
    if(signature.empty() || signature[0] != '(') {
      throw std::invalid_argument(signature.data());
    }

    size_t p = signature.find(')');
    if(p==1) {
      i_dim[i] = 0;
    } else if(p==2) {
      char dim = signature[1];
      for(i_dim[i]=0; i_dim[i]<n_dims; i_dim[i]++) {
	if(dim_list[i_dim[i]] == dim) break;
      }
      if(i_dim[i] == n_dims) {
	dim_list[n_dims] = dim;
	n_dims++;
      }
      
      i_dim[i]++;
    } else {
      throw std::invalid_argument(signature.data());
    }
    
    p = signature.find('(', p+1);
    if(p>signature.size()) {
      break;
    }
    signature.remove_prefix(p);
  }

  return i_dim;
}

// Get type signature of function
template<typename... T>
constexpr const char* get_types(void(*f)(T...))
{ return (const char[]) { arg_info<T>::dtype_char... }; }

// Get number of inputs from signature
constexpr size_t get_nin(std::string_view signature) {
  size_t nin = 0;
  size_t p1 = signature.find("->");
  size_t p2 = signature.find("(");
  while(p2<p1) {
    nin++;
    p2 = signature.find("(", p2+1);
  }
  return nin;
}

struct ufunc_type_implementation {
  PyUFuncGenericFunction fFunc;
  const char* fSignature;
  const char* fTypes;
  size_t fNargs;
  size_t fNin;
};

// Pre-processor macro to register a ufunc implementation
#define add_ufunc_impl(ufunc_impl, f_align, f_unalign, signature)	\
  static ufunc_type_implementation ufunc_impl = {			\
    [](char** args, const npy_intp* dims, const npy_intp* steps, void*){ execute_ufunc(f_align, f_unalign, parse_sig(signature), args, dims, steps); }, \
    signature,								\
    get_types(f_align),							\
    get_nargs(f_align),							\
    get_nin(signature)							\
    };
 


// Collect list of ufunc implementations as a single ufunc
struct ufunc_implementation {
  size_t fN;
  PyUFuncGenericFunction* fFuncs;
  char* fTypeSigs;
  size_t fNargs, fNin, fNout;
  const char* fName;
  const char* fSignature;
  const char* fDescription;

  ufunc_implementation(const std::vector<ufunc_type_implementation >& impl_list, const char* name, const char* description) :
    fN(0), fFuncs(NULL), fTypeSigs(NULL), fNargs(0), fNin(0), fNout(0),
    fName(name), fSignature(NULL), fDescription(description) {
    for(auto impl : impl_list) {
      if(fSignature==NULL) fSignature = impl.fSignature;
      if(fNargs==0) fNargs = impl.fNargs;
      if(fNin==0) fNin = impl.fNin;
      if(fNout==0) fNout = fNargs - fNin;
      // Make sure all functions have the same signature!
      assert(fSignature==impl.fSignature && fNargs==impl.fNargs && fNin==impl.fNin);
      
      if(fFuncs==NULL)
	fFuncs = new PyUFuncGenericFunction[impl_list.size()];
      fFuncs[fN] = impl.fFunc;
      
      if(fTypeSigs==NULL)
	fTypeSigs = new char[fNargs*impl_list.size()];
      strncpy(fTypeSigs + fN*fNargs, impl.fTypes, fNargs);
      fN++;
    }
  }
};

#define create_ufunc(ufunc_var, ufunc_name, ufunc_doc, ...)	\
  static ufunc_implementation ufunc_var({ __VA_ARGS__ },		\
					ufunc_name,			\
					ufunc_doc );



#endif
