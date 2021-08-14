# This file is a simple discretized Lorenz system with visualization
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L4Environment")

# define Lorenz dynamics
function lorenz(u,p)
  α,σ,ρ,β = p
  du1 = u[1] + α*(σ*(u[2]-u[1]))
  du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
  du3 = u[3] + α*(u[1]*u[2] - β*u[3])
  [du1,du2,du3]
end
p = (0.02,10.0,28.0,8/3)

# solve system with dynamics function and save output
function solve_system_save(f,u0,p,n)
  u = Vector{typeof(u0)}(undef,n)
  u[1] = u0
  for i in 1:n-1
    u[i+1] = f(u[i],p)
  end
  u
end
to_plot = solve_system_save(lorenz,[1.0,0.0,0.0],p,1000)

# plot output using plots
using Plots
x = [to_plot[i][1] for i in 1:length(to_plot)]
y = [to_plot[i][2] for i in 1:length(to_plot)]
z = [to_plot[i][3] for i in 1:length(to_plot)]
plot(x,y,z)
