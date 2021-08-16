# This file implements a multithreaded parameter search
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L5Environment")

using StaticArrays, BenchmarkTools
# statically allocated dynamics function solver
function lorenz(u,p)
  α,σ,ρ,β = p
  @inbounds begin
    du1 = u[1] + α*(σ*(u[2]-u[1]))
    du2 = u[2] + α*(u[1]*(ρ-u[3]) - u[2])
    du3 = u[3] + α*(u[1]*u[2] - β*u[3])
  end
  @SVector [du1,du2,du3]
end

# in-place trajectory solver
function solve_system_save!(u,f,u0,p,n)
  @inbounds u[1] = u0
  @inbounds for i in 1:length(u)-1
    u[i+1] = f(u[i],p)
  end
  u
end

p = (0.02,10.0,28.0,8/3)
u = Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef,1000)
# @btime solve_system_save!(u,lorenz,@SVector([1.0,0.0,0.0]),p,1000)

# in-place statically allocated mean trajectory function, one for each thread
const _u_cache_threads = [Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef,1000) for i in 1:Threads.nthreads()]
function compute_trajectory_mean4(u0,p)
    solve_system_save!(_u_cache_threads[Threads.threadid()],lorenz,u0,p,1000);
    mean(_u_cache_threads[Threads.threadid()])
end

# generate random sets of parameters to search over
ps = [(0.02,10.0,28.0,8/3) .* (1.0,rand(3)...) for i in 1:1000]

# conduct parameter search serially over sets of parameters
serial_out = map(p -> compute_trajectory_mean4(@SVector([1.0,0.0,0.0]),p),ps)
# conduct parameter search multithreaded
using Base.Threads
function tmap(f,ps)
  out = Vector{typeof(@SVector([1.0,0.0,0.0]))}(undef,1000)
  Threads.@threads for i in 1:1000
    out[i] = f(ps[i])
  end
  out
end
threaded_out = tmap(p -> compute_trajectory_mean4(@SVector([1.0,0.0,0.0]),p),ps)

@btime serial_out = map(p -> compute_trajectory_mean4(@SVector([1.0,0.0,0.0]),p),ps)
@btime threaded_out = tmap(p -> compute_trajectory_mean4(@SVector([1.0,0.0,0.0]),p),ps)
