# This file implements a barebones neural network
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L5Environment")

# make an asynchronous call to threads and wait for all to resolve
using Distributed
@time begin
    a = Vector{Any}(undef,nworkers())
    @sync for (idx, pid) in enumerate(workers())
        @async a[idx] = remotecall_fetch(sleep, pid, 4)
    end
end
