## Types of Parallelism
08/22/21

In this lecture, we will start constructing parallel equations for integration of scientific models:
```
u' = f(u,p,t) => u = int(f, t0, tf)du + u0
```

### Solving ODEs in Julia
We will use the Lorenz equation as an illustrative example, with the in-place version shown here:

```
function lorenz(du,u,p,t)
 du[1] = p[1]*(u[2]-u[1])
 du[2] = u[1]*(p[2]-u[3]) - u[2]
 du[3] = u[1]*u[2] - p[3]*u[3]
end
```

This is an implementation of the following system of three variables:

1. dx/dt = &sigma;(y - x)
2. dy/dt = x(&rho; - z) - y
3. dz/dt = xy - &beta;z

Then, to integrate this equation using Julia's `DifferentialEquations`:

```
u0 = [1.0,0.0,0.0]
tspan = (0.0,100.0)
p = (10.0,28.0,8/3)

using DifferentialEquations
prob = ODEProblem(lorenz,u0,tspan,p)
sol = solve(prob)
```

The solution to this differential equation is given as a continuous solution, equipped with an interpolation method that allows sampling at any point

### "Hard" ODE Integration Problems
Biochemical equations often have large separation of timescales, which makes them tougher to solve (which is expanded upon later). Robertson equation is an illustrative example:

```
using Sundials, ParameterizedFunctions
function rober(du,u,p,t)
  y₁,y₂,y₃ = u
  k₁,k₂,k₃ = p
  du[1] = -k₁*y₁+k₃*y₂*y₃
  du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
  du[3] =  k₂*y₂^2
end
```

Integration is performed using the stiff integration method `Rosenbrock23`, which is designed for stiff ODE problems.

```
prob = ODEProblem(rober,[1.0,0.0,0.0],(0.0,1e5),(0.04,3e7,1e4))
sol = solve(prob,Rosenbrock23())
plot(sol)
```

### Geometric Properties
#### Linear Systems
We will use the scalar, linear ODE system as a test problem to evaluate stability: **u' = &alpha;u** => **u(t) = u(0) • exp(&alpha;t)**

There are three different outcomes of an ODE system of this type:
1. Re(&alpha;) > 0 => u(t) -> infinity as t -> infinity
2. Re(&alpha;) < 0 => u(t) -> 0 as t -> infinity
3. Re(&alpha;) = 0 => u(t) is periodic or constant as t -> infinity

This can be extended to nonlinear systems, where these conditions are placed over all updates for the system after eigendecomp on the dynamics matrix (z' = D • z)

1. Re(Di) < 0 for all Di => u(t) -> 0 as t -> infinity
2. Re(Di) > 0 for any Di => u(t) -> infinity as t -> infinity

#### Nonlinear Systems
For continuous derivatives the above applies locally (using the locally linear approximation for continuous differential systems)

### Numerically Solving ODE Systems
The below is the Explicit Euler method for solving differential equations, which corresponds to the update form of the discrete linear system:

**f(u,p,t) = u' = du/dt ~ &delta;u/&delta;t** => **&delta;t • f(u,p,t) = u(n+1) - u(n)**

To understand the error incurred with this method, we look at the Taylor expansion around **t**:

**u(t + &delta;t) = u(t) + &delta;t • u'(t) + &delta;t^2 • 1/2 • u''(t)**

If we truncate this solution at the first derivative, you can see that we incur an estimation error of O(&delta;t^2). In order to construct a method which estimate error to higher orders of accuracy, we can expand this expansion and estimate using for function calls to our derivative function. Common derivative orders are 4 or 8 (i.e., RK4 method). These solution routines are commonly implemented in more low-level programming languages such as C/Fortran and called from the normal scripting language (for efficiency).

The following are some benchmarks across languages for the relationship between speed and error in explicit integration methods:
[](./IntegrationBenchmarks.png)

### Stability of Numerical Integration Methods
Order of error from truncation is not the full story of an integration method, since even if the error generated from one timestep is small, it's possible that these errors will not dissipate (or even exponentially grow!) with further timesteps. This is known as the **stability** of the integration method with respect to timestep. The test system that we will use to evaluate this phenomena is the scalar linear ODE system: **u' = &alpha; • u**

Important to remember is that stability is a local phenomenon, and so we can't necessarily understand stability of an ODE in a global sense, especially for multivariable or nonlinear systems. There are ways to charactertize stability in these systems, but it is using other metrics than these. 

#### Explicit Euler
For the explicit Euler formulation, the stability over timesteps can be considered using the following equation:

**u(n+1) = (1 + z) • u(n)** where **z = &delta;t • &alpha;**

It's easy to see that, with this formulation, if **(1+z) > 1**, errors will not dissipate, and actually will exponentially grow with more timesteps. Even if &alpha; is less than 1 (and so the system converges on 0 as t -> infinity), if the timestep is too large, the solution will explode rapidly

#### Implicit Euler
For the implicit (backward) Euler formulation, stability over timesteps can be expressed as a similar way to above:

**u(n+1) = (1/(1-z)) • u(n)** where **z** is the same as above

With the implicit Euler method, errors dissipate when **(1/(1-z)) < 1** => unstable if **0 < z < 2**. This method is both L-stable and A-stable wrt **z**

- **L-stability**: stable for all Re(z) < 0
- **A-stability**: stable as z -> infinity

### Stiffness and Timescale Separation
[](./Stiffness.png)

The combination of short and long timescale dynamics can cause integration to diverge when only long timescale dynamics are desired. In other words, if we care about timescales on the order of 10 minutes, but there are short term oscillations on the order of milliseconds, long timesteps with explicit methods can cause integration to diverge for timesteps far smaller than the scale of the desired dynamics

### Poincaré-Bendixson Theorem
This theorem states that there are only 3 possibilities for asymptotic values in a (nice) ODE system:
1. Steady-state
2. Diverges
3. Oscillates on a periodic orbit
