# c_printf
`c_printf` wraps `printf(<format>, ...)` to include color specification for each parameter. The main idea of the API is keep the printf structure, but with a small extension.

## Usage
Examples about how to use it:
```
c_printf("[r]%s", "red text");
c_printf("[g]%s", "green text");
c_printf("[r]%s [g]%s", "red", "green");
c_printf("[y]%d", 52334);
c_printf("[r]%s %s", "red", "normal");
```

## Available Colors
Each color is specified with its first character (in lowercase, i.e. `[r]` for red). The following are currently available:

* RED = `[r]`
* GREEN = `[g]`
* YELLOW = `[y]`
* BLUE = `[b]`
* MAGENTA = `[m]`
* CYAN = `[c]`
* NORMAL = `[n]`
 
## Installation
Just include it on you file as always `#include <c_printf.h>` to import necessary symbols. You can install using `clib` package manager:

```
$ clib install gioyik/c_printf
```
