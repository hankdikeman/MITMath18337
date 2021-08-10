# this file implements an out-of-place statically allocated array DiffEq solver for the Henon-Heiles system
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L2Environment")

# out of place static solve
using StaticArrays
function HH_OOP_STATIC(u, p, t)
    dx = u[2]
    dpx = -u[1] - 2*u[1]*u[3]
    dy = u[4]
    dpy = -u[3] - (u[1]^2 - u[3]^2)
    return @SVector [dx, dpx, dy, dpy]
end

# benchmark out of place dynamically allocated solution
using DifferentialEquations, BenchmarkTools
u0 = @SVector [0.1,0.0,0.1,0.0]
tspan = (0.0,100.0)
prob = ODEProblem(HH_OOP_STATIC,u0,tspan)

# print out benchmark results
println(@benchmark solve(prob,Tsit5()))
