# This file implements a barebones neural network
# Author:   Henry Dikeman
# Contact:  dikem003@umn.edu

# activate environment
cd(@__DIR__)
using Pkg
Pkg.activate("./L3Environment")

using DifferentialEquations, Plots, Flux
k = 1.0
force(dx,x,k,t) = -k*x + 0.1sin(x)
prob = SecondOrderODEProblem(force,1.0,0.0,(0.0,10.0),k)
sol = solve(prob)
# plot(sol,label=["Velocity" "Position"])

# force as a function of position for many points (~1000)
plot_t = 0:0.01:10
data_plot = sol(plot_t)
positions_plot = [state[2] for state in data_plot]
force_plot = [force(state[1],state[2],k,0) for state in data_plot]

# Generate the dataset
t = 0:3.3:10
dataset = sol(t)
position_data = [state[2] for state in sol(t)]
force_data = [force(state[1],state[2],k,t) for state in sol(t)]

# plot(plot_t,force_plot,xlabel="t",label="True Force")
# scatter!(t,force_data,label="Force Measurements")

# define short NN to estimate force
NNForce = Chain(x -> [x], # push to scalar array
                   Dense(1, 32, tanh),
                   Dense(32, 1),
                   first)
# encode loss function as how well we fit to datapoints
loss() = sum(abs2,NNForce(position_data[i]) - force_data[i] for i in 1:length(position_data))
# second loss function that we satisfy Hooke's Law
random_positions = [2rand()-1 for i in 1:100] # random values in [-1,1]
loss_ode() = sum(abs2,NNForce(x) - (-k*x) for x in random_positions)

# combine loss functions
λ = 0.1
composed_loss() = loss() + λ*loss_ode()

# OLD TRAINING METHOD, NO REGULARIZATION
# train with SGD
# opt = Flux.Descent(0.01)
# data = Iterators.repeated((), 5000)
# iter = 0
# cb = function () #callback function to observe training
  # global iter += 1
  # if iter % 500 == 0
    # display(loss())
  # end
# end
# display(loss())
# Flux.train!(loss, Flux.params(NNForce), data, opt; cb=cb)

# NEW TRAINING METHOD, INCLUDE ODE REGULARIZATION
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(composed_loss())
  end
end
display(composed_loss())
Flux.train!(composed_loss, Flux.params(NNForce), data, opt; cb=cb)

# plot training against given datapoints
learned_force_plot = NNForce.(positions_plot)

plot(plot_t,force_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_force_plot,label="Predicted Force")
scatter!(t,force_data,label="Force Measurements")
