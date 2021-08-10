# this file implements an in-place dynamically allocated array DiffEq solver for the Henon-Heiles system
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L2Environment")

# out of place dynamic solve
function HH_IP!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1] - 2*u[1]*u[3]
    du[3] = u[4]
    du[4] = -u[3] - (u[1]^2 - u[3]^2)
end

# benchmark out of place dynamically allocated solution
using DifferentialEquations, BenchmarkTools
u0 = [0.1;0.0;0.1;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(HH_IP!,u0,tspan)

# print out benchmark results
println(@benchmark solve(prob,Tsit5()))
