#include "eigen_ufunc.hh"

#include "convolve.hh"
#include "derivative.hh"
#include "fft.hh"
#include "mean.hh"
#include "pole_zero.hh"
#include "trap_filter.hh"
#include "trap_norm.hh"

// Preprocessor directives used to set the module name using -Dmodule_name=...
#define str(A) xstr(A)
#define xstr(A) #A
#define PyInit(A) xPyInit(A)
#define xPyInit(A) PyInit_##A(void)

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  str(module_name),
  NULL,
  -1,
  NULL,
  NULL,
  NULL,
  NULL,			
  NULL
};
static void *data[1] = { NULL };
  
PyMODINIT_FUNC PyInit(module_name) {

  // ADD NEW UFUNCS HERE
  std::vector<ufunc_implementation> ufunc_list = {
    convolve_full_ufunc,
    convolve_same_ufunc,
    convolve_valid_ufunc,
    convolve_ufunc,
    //    derivative_ufunc,
    fft_convolve_ufunc,
    fft_ufunc,
    mean_ufunc,
    pole_zero_ufunc,
    trap_filter_ufunc,
    trap_norm_ufunc
  };
  
  
  PyObject* m = PyModule_Create(&moduledef);
  if(!m) return NULL;
  
  import_array();
  import_umath();
  
  PyObject* d = PyModule_GetDict(m);
  
  for(ufunc_implementation& ufunc : ufunc_list ) {
    PyObject* ufunc_obj = PyUFunc_FromFuncAndDataAndSignature(
      ufunc.fFuncs,
      data,
      ufunc.fTypeSigs,
      ufunc.fN,
      ufunc.fNin,
      ufunc.fNout,
      PyUFunc_None,
      ufunc.fName,
      ufunc.fDescription,
      0,
      ufunc.fSignature
    );
    PyDict_SetItemString(d, ufunc.fName, ufunc_obj);
    Py_DECREF(ufunc_obj);
    }

    return m;
  }
