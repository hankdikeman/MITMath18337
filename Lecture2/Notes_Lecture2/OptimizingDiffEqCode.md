## Optimizing DiffEq Programs in Julia
08/08/21

Models of dynamic systems are representative of many scientific computing systems: dynamic systems accept a state (and potentially a list of parameters) and output a rate. This simple function is typically called many times, can be several for each timestep with dynamic timestepping algorithms or implicit solvers, so an efficient implementation of the RHS of your ODE or discretized PDE system is the best way to optimize your system

### Optimizing small systems
For systems where the number of independent variables are very small (think <10), much of the cost of a function call is accounted for by repeated allocations. Even for allocations of small chunks of memory, each allocation can be relatively expensive compared to most arithmetic operations. The different level of optimizations are listed below in terms of speed:
1. **Out-of-place dynamic allocation**: when small chunks of memory in the heap must be allocated on each function call (say, for instance, to allocate memory for derivative values) the time-complexity of a function call can increase dramatically. This is the worst way to implement a function such as the RHS of an ODE system in terms of speed
2. **In-place dynamic allocation**: by precaching memory locations to contain the RHS of your ODE system, this allocation must only be performed once. Then, in later function calls, this memory is simply written over "in-place". This is far faster than repeated allocations (and probably fine for most systems)
3. **Out-of-place static allocation**: while IP and OOP dynamic allocation store memory on the heap, a static allocation allows memory (of size determined at compile time) to be stored on the stack. Memory on the stack can be allocated and accessed at essentially zero cost, which means this is the best approach for small systems. However, stack memory is far more limited, and there are consequences to allocating too much stack memory for variables, so this can only be used for small systems

### Optimizing Large Systems
The above principles also apply to large systems, although static allocation typically will not work for large systems. The best way to optimize a large system is to do the following:
1. Build the simplest (probably slowest) implementation
2. Remove allocations due to memory copies if they occur
3. Cache any heap allocated variables that are created within the function
4. Perform all operations in-place when possible, avoid vectorization
5. Implement broadcasting across consecutive operations, prevent temporary allocations
6. Use `@inbounds` macro to prevent bounds checking (when safe)
7. As a last resort, devectorize all operations and instead write out each matrix operation (only in Julia)

Biggest take home point: **dynamic memory allocations are poison to efficient code, prevent as many of these as you can and iteratively trade-off program complexity for efficiency by eliminating memory allocation**
