## Optimizing Serial Code (in Julia)
08/08/21

Before parallelization, the serial version of your program must be optimized (otherwise parallel programming won't help much). This lecture teaches some of the basics of serial optimization, connecting them to the fast and slow operations in relation to memory allocation (dynamic vs static), control flow, and @jit compiler optimization

### Computer Memory Structure
[Memory Structure](./GraphicsFromLecture/CPUMemoryStructure)
Memory structure is divided into separate intermediary caches separate from the registers present in the CPU core. Obviously, the slowest access is in main memory, and the fastest is any registers available on the CPU core (which may be hundreds or more) but which are not usually used for memory storage. In order of speed, these intermediary caches are as follows:
1. **L1 Cache**: most immediately available to the CPU, but not shared between CPUs
2. **L2 Cache**: second fastest, available to all CPUs (each CPU can access memory in a shared L2 cache)
3. **L3 Cache**: slowest, mostly responsible for loading data from main memory

If memory is not preloaded into one of these caches and is required by the CPU, this is called a "cache miss" and can take several orders of magnitude longer than loading from the L1 cache, and significantly longer than loading from the L2 or L3 cache.

Since cache storage is not directly controlled by the programmer for languages like Julia and Python, it's a good idea to deal with data in the order that the language expects you to pull it. For arrays:
- **Julia/MatLab/Fortran**: expect you to deal with data *columnwise*
- **Python**: expects you to deal with data *rowwise*

### Program-Level Memory Structure
- **stack**: statically-assigned memory. Memory for which the size is known *at compile time*. Allocated as chunks of direct memory
- **heap**: dynamically-assigned memory. Memory for which the size is known *at runtime*. Heap is allocated dynamically as sets of pointers to fixed chunks of memory

*hint*: to see the allocations inherent in a set of Julia operations, run function call with `@btime` allocator

### Broadcasting in Julia
Large savings can be gained by avoiding the dynamic allocation inherent in storing intermediate results (effectively by using inplace operations). The difference between `sin(A)` and `sin.(A)` is that `sin(A)` allocates an intermediate array, whereas `sin.(A)` is directly called on the data within `A`.

A related topic is views vs copies in Julia. By default, slicing an array will create a copy of the memory at the address of the sliced array, meaning that changes to the slices section of the array (the new variable) will not change the original data. In constrast, using an array view (very similar to the PyTorch implementation) with the `@view` operator will change the underlying data.

### More on Dynamic (Heap) Allocations
While stack allocations are static, with O(*1*) time complexity and a small constant, dynamic heap allocations tend to be O(*n*) time complexity with a large constant. In generate, heap allocations will require ~100 times more clock cycles that their static allocation counterpart. See: the following infographic for a useful heuristic

[Approximate Runttime](./GraphicsFromLecture/ApproxClockCycles)

What's worse is that if heap memory is exceeded (i.e., you run out of RAM), your computer will begin using *swap* memory, which is far slower than RAM (sometimes so slow as to make the program appear to be hung). For this reason, when optimizing, it is often far better to preallocate memory (or use an existing memory structure) to avoid dynamic allocations.

### Memory Optimization Summary
- Avoid cache misses when possible
- Iterate column-wise (to reduce cache misses)
- Avoid heap allocations when allocations can be handled by the stack
- Loops aren't slower than vectorization in Julia (for reasons explained later)

### Why is Julia faster than Python (in general)?
1. Powerful type inference engine
2. Type specialization in functions

Point-by-point, this means the following:
1. **Type Inference**: Python *always* uses heap allocation, since all data is allocated at runtime rather than compile time. Julia instead evaluates types before executing code, which permits fast stack allocation rather than being reduced to pointer allocation for all memory
2. **Function Type Specialization**: Julia prechecks the types moving in/out of functions and can optimization compilation to reduce type checking, allowing "generic" functions to be changed to their specific counterparts at compile-time. 

**Further Explanation of #2 Above**: Using the theoretical function `ff` as an example, we can define the function `ff(x,y)`, which is functionally equivalent to `ff(x::Any, y::Any)`, and cannot be optimized much since the function *does not know* what type will be input to the function. Thus, the suboperations of the function (such as `ADD` operations) must be changed to the relevant assembly code at runtime using assembly level control flow. It may be desired to generate multiple implementation for specific types that can be subcontracted out at compile time:
1. `ff(x::Float64, y::Float64)`
2. `ff(x::Int64, y::Int64)`
3. `ff(x::Number, y::Number)`

Thus, the first two will be used for `Float64` or `Int64` operations, and the final is a fallback function that can be used for all others. Thus, each function can skip a lot of the dynamic typechecks (except the fallback function) which allows more seamless type inference. 


The above compilation protocol is very efficient and robust, but certain operations can break the performance of these protocols. These are detailed below...

### Untyped Containers
When arrays (or matrices) are passed as arguments to a function with no type information, these can break the type inference engine since Julia cannot know at compile time whether an array will contain multiple types of data or not. For example, the following function will run very slow:
```
function r(x)
    .
    .
    .
    for ...
        c = f(x[1])
        d = ff(x[2], c)
    end
    return d
end
```
This is because for every iteration of the inner `for` loop, memory locations will be dynamically allocated for `x[1]` and `x[2]`, which is a lot of overhead if the for loop is long with relatively simple operations inside the loop.

A much faster implementation would be to preallocate those element variables, then pass those as parameters to the function:
```
s(x) = _s(x[1], x[2])
function _s(x1, x2)
    .
    .
    .
    for ...
        c = f(x1)
        d = ff(x2, c)
    end
    return d
end
```
Thus, the variables are dynamically allocated **once** when the function is called, and not repeatedly within the inner loop. This can result in a speedup of a factor of several hundred if the dynamic allocation is the lion's share of the time complexity (as is often the case for relatively simple functions).

A related issue is with the allocation of globals. If the above function was performed using any global variables, type inference would also break because the engine within Julia would be unable to know if a global variable could be suddenly changed anywhere in the program (which would prevent blind execution).

### Final Notes on Overhead Analysis
Julia still has extensive bounds checks on array and matrix operations, since attempting to read or write from "out-of-bounds" memory can be dangerous, and result in Segmentation Faults are you not careful. However, you can still overwrite this bounds checking as an optimization by placing the `@inbounds` macro on matrix reads and writes. The reason this can make a larger difference than simply trimming the bounds-checking is that modern CPU's rely heavily on SIMD speedups (*Single Input Multiple Data*), where certain CPU operations are lumped together. This certainly won't work if each CPU operation is interrupted by a bounds-check. For this reason, this can be a useful optimization if used carefully)
