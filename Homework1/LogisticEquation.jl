# Newton iteration method for finding the quantile of an arbitrary continuous distribution
cd(@__DIR__)
using Pkg
Pkg.activate("./H1Environment")

function logistic!(x::Float64, r::Float64)
    # calculate next value in-place
    x = r * x * (1-x)
    return x
end

function calc_attractor!(out::AbstractArray, f, p::Float64, num_attract::Int64=150, warmup::Int64=400, x0::Float64=0.25)
    # perform warmup to arrive at steady state
    for i in 0:warmup
        x0 = f(x0, p)
    end

    # save remaining values
    for i in 1:num_attract
        x0 = f(x0, p)
        @inbounds out[i] = x0
    end

    # return filled array
    return out
end

paramvector = [x for x in 2.9:0.001:4.0]
outmat = zeros(Float64, 150, length(paramvector))

function logistic_param_search!(outmat::AbstractArray, paramvector::AbstractArray)
    parameter_count = 1
    @inbounds begin
        for param in paramvector
            calc_attractor!(view(outmat,:,parameter_count), logistic!, param)
            parameter_count += 1
        end
    end
end

using Base.Threads
function mthread_logistic_param_search!(outmat::AbstractArray, paramvector::AbstractArray)
    Threads.@threads for i in 1:length(paramvector)
        @inbounds begin
            calc_attractor!(view(outmat,:,i), logistic!, paramvector[i])
        end
    end
end

println("thread num: $(Threads.nthreads())")

using BenchmarkTools
println("serial implementation:")
logistic_param_search!(outmat, paramvector)
@btime logistic_param_search!(outmat, paramvector)

println("multithreaded implementation:")
mthread_logistic_param_search!(outmat, paramvector)
@btime mthread_logistic_param_search!(outmat, paramvector)

using Plots
plot(paramvector, transpose(outmat), linestyle=:dot, seriescolor=:black, legend=false, markersize=0.1, seriesalpha=0.02, xlabel = "parameter value (r)", ylabel = "steady state value (x)", title = "Logistic Equation Steady State Dynamics")
