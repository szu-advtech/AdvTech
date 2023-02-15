///  common.h
///  This header is to include common macros, exceptions, helpers etc.

#ifndef __COMMON_H__
#define __COMMON_H__
 
#include <string>
#include "compatible.h"

/// Safely delete a pointer to an object
#ifndef SAFE_DELETE
#  define SAFE_DELETE(ptr) if(ptr != NULL) \
                        { delete ptr; ptr = NULL; }
#endif

/// Safely delete a pointer to an array of objects
#ifndef SAFE_DELETE_ARRAY
#  define SAFE_DELETE_ARRAY(ptr) if(ptr != NULL) \
                        { delete[] ptr; ptr = NULL; }
#endif

/// Common data type

#ifndef LARGE_INT64
typedef long long int __INT64__;
#define LARGE_INT64  __INT64__
#endif

#include "mi_util.h"
#include "mi_limit.h"

#endif //:~ __COMMON_H__
