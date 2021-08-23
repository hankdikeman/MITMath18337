# A simple test of the CUDA Julia wrapper (array and kernel-based)
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L6Environment")

# import CUDA
using CUDA

# array-based
# define matrices
A = rand(100, 100); B = rand(100, 100)
# move to GPU
cuA = cu(A); cuB = cu(B)
# multiply on GPU
cuC = cuA * cuB
# move to CPU
@show C = Array(cuC)

# kernel-based
# create tensors
N = 2^20
x_d = CUDA.fill(1.0f0, N)
y_d = CUDA.fill(2.0f0, N)

# define in-place GPU addition
function gpu_add2!(y, x)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

# use function
fill!(y_d, 2)
@cuda threads=256 gpu_add2!(y_d, x_d)
@show all(Array(y_d) .== 3.0f0) # evaluates to True
