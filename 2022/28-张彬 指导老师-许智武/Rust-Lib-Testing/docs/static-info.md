# Static infomation

----------

### Through the function signature:
- API dependency graph with parameter type and the type of return value --> producers and consumers
- Transfer of ownership (move/borrowing)
- unsafe function
- generic type, trait object, closures


### Through the static analysis based on MIR:
- Call graph --> function coverage 
- Internal unsafe --> interior mutability
- Get raw pointer
- Uninitialized value
- lock
- def-use chain


### Through the running/analysis of the fuzz target
- compiler-inserted invisible code (*e.g.*, *Drop()*)
- branch coverage
- compilation error --> refine 