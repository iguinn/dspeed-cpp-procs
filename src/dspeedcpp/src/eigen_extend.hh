// Use this to overload operator[] to return a column

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
ColXpr operator[](Index i)
{
  return ColXpr(derived(), i);
}

/// This is the const version of col().
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
ConstColXpr operator[](Index i) const
{
  return ConstColXpr(derived(), i);
}
  
