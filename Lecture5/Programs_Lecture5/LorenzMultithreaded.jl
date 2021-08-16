# This file implements a barebones neural network
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L5Environment")

using StaticArrays, BenchmarkTools
# define dynamics function
function lorenz!(du,u,p)
  α,σ,ρ,β = p
  @inbounds begin
    du[1] = u[1] + α*(σ*(u[2]-u[1]))
    du[2] = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du[3] = u[3] + α*(u[1]*u[2] - β*u[3])
  end
end

using Base.Threads
function lorenz_mt!(du,u,p)
  α,σ,ρ,β = p
  let du=du, u=u, p=p
    Threads.@threads for i in 1:3
      @inbounds begin
        if i == 1
          du[1] = u[1] + α*(σ*(u[2]-u[1]))
        elseif i == 2
          du[2] = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
        else
          du[3] = u[3] + α*(u[1]*u[2] - β*u[3])
        end
        nothing
      end
    end
  end
  nothing
end

# define main solution loop
function solve_system_save_iip!(u,f,u0,p,n)
  @inbounds u[1] = u0
  @inbounds for i in 1:length(u)-1
    f(u[i+1],u[i],p)
  end
  u
end

# solve system for 1000 timepoints
p = (0.02,10.0,28.0,8/3)
u = [Vector{Float64}(undef,3) for i in 1:1000]
@btime solve_system_save_iip!(u,lorenz!,[1.0,0.0,0.0],p,1000)
