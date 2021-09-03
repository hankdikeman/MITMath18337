# a "hello world" case for MPI programming using MPI.jl wrapper
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L12Environment")
Pkg.instantiate()

using MPI

MPI.Init()

# get MPI process rank id
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# get number of MPI processes in communicator
nproc = MPI.Comm_size(MPI.COMM_WORLD)

print("Hello world, I am rank $(rank) of $(nproc) processors\n")

MPI.Barrier(MPI.COMM_WORLD)
