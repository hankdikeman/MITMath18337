## Parameter Estimation and Reverse-Mode Autodiff
08/29/21

The last discussion was how to compute gradients quickly with the "adjoint" or "backpropagation" method, using computational graphs. In this lecture we will flesh out the remainder of a backpropagation framework, which requires the following:
1. A way of implementing autodifferentiation on a language
2. A systematic way to compute pullback quantities
3. *Any better ways to perform backpropagation on entire models*

### Implementation of Reverse-mode Autodifferentiation
It was relatively simple to implement forward-mode autodiff with dual (and multidual) arithmetic and simple operational rules, but **reverse-mode autodiff** is more complex, since it requires unrolling all the operations within a program from back to front, which is more difficult.

### Static Graph Autodiff
The simplest form of RM-autodiff is performed by defining a static, unchanging compute graph with precompiled gradient functions that are calculated using pullback. `Tensorflow` is one such framework. This paradigm has some limitations, such as the fact that you need to rewrite ALL existing functions to this structure, and you can't adapt existing functions to this architecture

### Tracing-based Autodiff and Wengert Lists
An alternative approach to static graph AD that can be used for composed functions are **Wengert Lists**, which uses repeated pullbacks to find update terms:

For function *f* with Jacobian *J*, *vT • J = (...((vT • JL)JL-1)...)J1* is performed, which is essentially repeated `jvp` applications

Since each value requires the previous pullback application, the following recursive process is used: *B(x,f,A) = B(x, f1, ...(B(fL-1:f1(x), fL, A))...)*

Thus, the pullback requires:
1. The operation performed
2. The value *x* of the forward-pass

The **Wengert List** is the trace of your forward pass that is "unwound" to find the *dC/dp* terms. This is also referred to as *tracing-based reverse-mode AD*. There are several frameworks that use this paradigm, including `PyTorch` and `ReverseDiff.jl`

#### Inspecting `Tracker.jl`
`Tracker.jl` is another *tracking-based RM autodiff* framework with a very simple implementation that we can inspect. The following is the implementation:

The `Call` struct stores a function and its argument
```
struct Call{F,As<:Tuple}
  func::F
  args::As
end
```

The `Tracked` function stores a node on the computational graph, that holds a function and its gradient
```
mutable struct Tracked{T}
  ref::UInt32
  f::Call
  isleaf::Bool
  grad::T
  Tracked{T}(f::Call) where T = new(0, f, false)
  Tracked{T}(f::Call, grad::T) where T = new(0, f, false, grad)
  Tracked{T}(f::Call{Nothing}, grad::T) where T = new(0, f, true, grad)
end
```

`TrackedReal` and `TrackedArray` store scalar and matrix-type parameter values
```
mutable struct TrackedReal{T<:Real} <: Real
  data::T
  tracker::Tracked{T}
end

struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  tracker::Tracked{A}
  data::A
  grad::A
  TrackedArray{T,N,A}(t::Tracked{A}, data::A) where {T,N,A} = new(t, data)
  TrackedArray{T,N,A}(t::Tracked{A}, data::A, grad::A) where {T,N,A} = new(t, data, grad)
end
```

This implementation does not necessarily inform you how to perform reverse differentiation through operations, such as a dot product. These reverse differentations are encoded through the `@grad` macro, shown here for a dot product:
```
@grad dot(xs, ys) = dot(data(xs), data(ys)), Δ -> (Δ .* ys, Δ .* xs)
```

This simple implementation shows the shortfall with tracing-based RM AD. Since structures are not strongly-typed, `Tracker.jl` and other implementations cannot precompile through the reverse pass:
- lots of heap allocations, requiring possible milliseconds of overhead for a small operation
- trace is value-dependent (on forward-pass) so precompiling is an issue
- if a trace is performed repeatedly (such as during training), the graph can be huge for naive implementations

There are some other implementations which directly compile operations "source-to-source", but this is limited to certain languages (generally precompiled or JIT languages) and is not a cure-all.

### Deriving Reverse-Mode Rules
In order to implement RM efficiently, we must be able to derive adjoint rules at a high level

#### Linear Solves
Imagine a system of the form *A • x = b*, which is defined by the parameterized system *A(p) • x = b(p)*. We want the gradients of our cost function *g(x,p)*:

*dg/dp = gp + gx • xp* => (adjoint)

While the derivatives with respect to the parameters (*gp* and *gx*) are relatively easy to find, but *xp*, the derivative of x with respect to p, is more difficult to find. We need to calculate this using the following:

*xpi = A^-1 • (bpi - Api • x)*

We can set *&lambda;T = gx • A^-1* and use this to simplify the equation (though not make it any more computationally efficient):

*dg/dp = gp - &lambda;T • (Ap • x - bp)*

This is common paradigm, where difficult-to-compute values are set to some value *&lambda;* and some simplification is performed

#### Nonlinear Solves (Newton's Method)
For some equation of the following:

*f(x, p) = 0*

Thus, we can define *df/dp* (the derivative of the function with respect to the parameters) as the following:

*df/dp = fx • xp + fp*

However, our goal is to get the derivative of our cost function *g* with respect to our parameters. This can be expressed as the following:

*dg/dp = gp + gx • xp* where *xp = -(fx^-1 • fp)*

This formulation involves building a term of size *M x MP* size, which is perhaps computationally intractable. Instead, we will again use the *&lambda;* Lagrangian Multiplier trick to solve for the following terms in turn:
1. *fxT • &lambda; = gxT*
2. *dg/dp = gp - &lambda;T • fp*

Which tells us how to change our parameters with respect to our error function *g*!

#### ODE Adjoint Method
If our forward pass involves integrating an ODE in time, we can also derive a method for changing our parameters with respect to the error in this integration:

*G(u,p) = G(u(p)) = integration from t0 to T: (g(u(t,p))dt)*

Through some signficant manipulation, and the following two definitions:
1. *&lambda;' = -(df/du) • &lambda; - (dg/du)*
2. *&lambda;(T) = 0*

This results in the following formulation:

*dG/dp = &lambda;(t0) • dG/du(t0) + int from t0 to T: &lambda;'dt*

Thus, the derivative of an ODE solution with respect to *g* is given by integrating simple derivative values backward in time, which can be used to calculate model updates with respect to the cost function, denoted as *dG/dp*

#### Complexities of Implementing ODE Adjoints
The whole problem of ODE adjoints can be boiled down to solving the following initial value problem:

*&lambda;' = -(df/du) • &lambda; - (dg/du)* with initial condition *&lambda;(T) = 0* from *T* to *t0*

But *df/du* is defined by *u(t)*, your solution at any point, which is only found on the forward pass. Thus, we need *u(t)* available at arbitrary *t* in order to compute the Jacobian with respect to the function anywhere. There are some popular approaches to tackling this computationally expensive question:
1. Solve *u' = f(u, p, t)* backwards in time, essentially appending *u(t)* to *&lambda;(t)* to the reverse solve. This can sometimes be numerically unstable, since values are not always the same when solving ODEs backwards in time due to floating point difficulty
2. Solve the forward ODE and get a continuous solution that can be interpolated to get *u* and *df/du* at any point needed. This is fast, but can be highly memory-intensive
3. Every time you need *u(t)*, re-integrate your ODE forward in time from your starting point to *u(t)*. This is very expensive, but can be improved by saving checkpoints of your ODE solution to integrate forward from (finitely many checkpoints, of course)

#### Connection to `NeuralODE` Methodology
If `f` in this case is a neural network, this approach reduces to the popular **neural ODE method**. The backward pass can be improved by recognizing that *(df/du) • &lambda;* is a single `vjp` calculate, with can be computed with a single backpropagation on `f` with the cost function at time *t* set to *&lambda;*: *C(t) = &lambda;*

This approach is highly valuable for fitting models in a data-driven matter to data, since it allows you to derive a model that encodes the physical rules of your ODE problem
