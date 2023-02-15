/// compatible.h
///  This header is used to make functions compatible across different platforms.

#ifndef __compatible_h__
#define __compatible_h__
 
#include <iostream>
#include <cmath>
#include <cassert>

#define COMPATIBILITY_OS_WIN     1

#if defined(_WIN32)
//define something for Windows (32-bit and 64-bit, this part is common)
#   define COMPATIBILITY_THIS_OS          COMPATIBILITY_OS_WIN
#endif
#   if defined(_WIN64)
//define something for Windows (64-bit only)
#       define COMPATIBILITY_THIS_OS_STR   "WIN64"
#   else
#       define COMPATIBILITY_THIS_OS_STR   "WIN32"
#   endif 

/// Check whether to use openmp
/// DO NOT FORGET to turn on "Open MP support"
///    [Visual studio]
///        Project properties / C/C++ / Open MP support: Yes (/openmp)
#if COMPATIBILITY_THIS_OS == COMPATIBILITY_OS_WIN
#   include <omp.h>
//  then _OPENMP will be defined
#else

#endif

#ifdef _OPENMP
#   define MI_USE_OMP
#   define MI_IS_OMP_USED   1
#else
// do not define MI_IS_OMP_USED
#   define MI_IS_OMP_USED   0
#endif

#endif /* __compatible_h__ */
