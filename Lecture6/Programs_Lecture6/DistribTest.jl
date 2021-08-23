# A simple test of the CUDA Julia wrapper (array and kernel-based)
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L6Environment")

# make an asynchronous call to threads and wait for all to resolve
using Distributed
addprocs(4)

# define global function
@everywhere f(x) = x .^ 2
t = Vector{Any}(undef,4)
xsq = zeros(4,10)

for i in 1:4
    t[i] = remotecall(f, i, randn(10))
end

for i in 1:4
    @show xsq[i,:] = fetch(t[i])
end
