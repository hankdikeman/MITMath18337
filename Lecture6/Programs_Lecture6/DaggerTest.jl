# A simple test of the CUDA Julia wrapper (array and kernel-based)
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L6Environment")

using Dagger

# define operations
add1(value) = value + 1; add2(value) = value + 2; combine(a...) = sum(a)

# define computational graph
p = delayed(add1)(4)
q = delayed(add2)(p)
s = delayed(combine)(p,q)

# collect value through computational graph
@show collect(s)
