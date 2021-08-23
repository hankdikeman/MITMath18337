# A simple test of the CUDA Julia wrapper (array and kernel-based)
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L6Environment")

using MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

dst = mod(rank+1, size)
src = mod(rank-1, size)

N = 4

send_mesg = Array{Float64}(undef, N)
recv_mesg = Array{Float64}(undef, N)

fill!(send_mesg, Float64(rank))

rreq = MPI.Irecv!(recv_mesg, src,  src+32, comm)

print("$rank: Sending   $rank -> $dst = $send_mesg\n")
sreq = MPI.Isend(send_mesg, dst, rank+32, comm)

stats = MPI.Waitall!([rreq, sreq])

print("$rank: Received $src -> $rank = $recv_mesg\n")

MPI.Barrier(comm)
