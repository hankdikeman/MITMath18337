# This file implements a barebones neural network
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L3Environment")

# weight and bias matrices
W = [randn(32,10),randn(32,32),randn(5,32)]
b = [zeros(32),zeros(32),zeros(5)]

# simple neural network
simpleNN(x) = W[3]*tanh.(W[2]*tanh.(W[1]*x + b[1]) + b[2]) + b[3]
simpleNN(rand(10))

# define library using Flux
using Flux
NN2 = Chain(Dense(10,32,tanh),
           Dense(32,32,tanh),
           Dense(32,5))
NN2(rand(10))

NN3 = Chain(Dense(10,32,x->x^2),
            Dense(32,32,x->max(0,x)),
            Dense(32,5))
NN3(rand(10))

# define neural network and loss function
NN = Chain(Dense(10,32),
           Dense(32,32),
           Dense(32,5))
# loss() = sum(abs2, sum(abs2, NN(rand(10)) .- 1) for i in 1:100)

# train neural network
# p = params(NN)
# Flux.train!(loss, p, Iterators.repeated((), 10000), ADAM(0.1))
# println(loss())

# approximating the function u` = cos(2πt)
NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
           Dense(1,32,tanh),
           Dense(32,1),
           first) # Take first value, i.e. return a scalar

# define PINN for u0 = 1
g(t) = t*NNODE(t) + 1f0

using Statistics
# define small ϵ for finite difference derivative with small ϵ to approximate derivative
ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ) - g(t)) / ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)

# perform training (with callbacks) for 5000 iterations
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    println(loss())
  end
end
println(loss())
Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)

# plot approximate against real solution
using Plots
t = 0:0.001:2.0
plot(t,g.(t),label="NN")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True Solution")





