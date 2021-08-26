# A simple implementation of a Newton fixed-point solver, which estimates derivatives using the ForwardDiff.jl autodiff engine
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L8Environment")

using ForwardDiff, StaticArrays

# define one newton step
function newton_step(f, x0)
    J = ForwardDiff.jacobian(f, x0)
    δ = J \ f(x0)
    return x0 - δ
end

# conduct 10 Newton steps, show intermediate and final results
function newton(f, x0)
    x = x0
    for i in 1:10
        x = newton_step(f, x)
        @show x
    end
    return x
end

# define nonlinear function and initial condition
ff(xx) = ( (x, y) = xx;  SVector(x^2 + y^2 - 1, x - y) )
x0 = SVector(10.0, 5.0)

# perform Newton iteration
x = newton(ff, x0)
