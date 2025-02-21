import numpy as np

def array(object, dtype=None, align: int = 64):
    """Create a memory-aligned numpy array with the contents of nda
    
    Parameters
    ----------
    object
        numpy array or array-like object
    align
        number of bytes to align to
    
    Returns
    -------
    aligned_array
        ndarray with copy of nda in aligned memory buffer
    """
    
    array_in = np.array(object, dtype=dtype, copy=False)
    array = empty(array_in.shape, array_in.dtype, align)
    np.copy_to(array_in, array)
    return array

def empty(shape, dtype=float, align: int = 64):
    """Create an empty memory-aligned column-major numpy array
    
    Parameters
    ----------
    shape
        numpy array shape
    dtype
        data type of array
    align
        number of bytes to align to
    
    Returns
    -------
    aligned_array
        empty ndarray
    """
    
    dtype = np.dtype(dtype)
    len = np.prod(shape)
    array = np.empty(len + align // dtype.itemsize, dtype=dtype)
    offset = ((align - array.ctypes.data)%align) // dtype.itemsize
    return array[offset : offset + len].reshape(shape, order='F')

def zeros(shape, dtype=float, align: int = 64):
    """Create a zero-filled memory-aligned column-major numpy array
    
    Parameters
    ----------
    shape
        numpy array shape
    dtype
        data type of array
    align
        number of bytes to align to
    
    Returns
    -------
    aligned_array
        empty ndarray
    """
    
    dtype = np.dtype(dtype)
    len = np.prod(shape)
    array = np.zeros(len + align // dtype.itemsize, dtype=dtype)
    offset = ((align - array.ctypes.data)%align) // dtype.itemsize
    return array[offset : offset + len].reshape(shape, order='F')

def ones(shape, dtype=float, align: int = 64):
    """Create a one-filled memory-aligned column-major numpy array
    
    Parameters
    ----------
    shape
        numpy array shape
    dtype
        data type of array
    align
        number of bytes to align to
    
    Returns
    -------
    aligned_array
        empty ndarray
    """
    
    dtype = np.dtype(dtype)
    len = np.prod(shape)
    array = np.ones(len + align // dtype.itemsize, dtype=dtype)
    offset = ((align - array.ctypes.data)%align) // dtype.itemsize
    return array[offset : offset + len].reshape(shape, order='F')


def full(shape, fill_value, dtype=None, align: int = 64):
    """Create a memory-aligned column-major numpy array with a repeating value
    
    Parameters
    ----------
    shape
        numpy array shape
    fill_value
        Fill value
    dtype
        data type of array
    align
        number of bytes to align to
    
    Returns
    -------
    aligned_array
        empty ndarray
    """
    
    dtype = np.dtype(dtype)
    len = np.prod(shape)
    array = np.full(len + align // dtype.itemsize, fill_value, dtype=dtype)
    offset = ((align - array.ctypes.data)%align) // dtype.itemsize
    return array[offset : offset + len].reshape(shape, order='F')
