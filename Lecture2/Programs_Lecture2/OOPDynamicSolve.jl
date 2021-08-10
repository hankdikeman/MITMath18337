# this file implements an out-of-place dynamically allocated array DiffEq solver for the Henon-Heiles system
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L2Environment")

# out of place dynamic solve
function HH_OOP(u, p, t)
    dx = u[2]
    dpx = -u[1] - 2*u[1]*u[3]
    dy = u[4]
    dpy = -u[3] - (u[1]^2 - u[3]^2)
    return [dx, dpx, dy, dpy]
end

# benchmark out of place dynamically allocated solution
using DifferentialEquations, BenchmarkTools
u0 = [0.1;0.0;0.1;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(HH_OOP,u0,tspan)

# print out benchmark results
println(@benchmark solve(prob,Tsit5()))
