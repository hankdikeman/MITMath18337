## Intro to Scientific Machine Learning
08/10/21

The basis of most physical models are dynamic systems, with equations describing the rates of change of given variables with respect to other variables. In this lecture:
1. Basic properties of discrete dynamical systems
2. Understanding stability in dynamical systems

### What is a discrete dynamical system?
A discrete dynamical system is defined by a relationship of the form: `u_n+1 = model(u_n, n)`, where a future state of the system is described by an initial state and a series of discrete updates

*Example 1:* an autoregression model called AR1 is defined by the relationship **u(n+1) = &alpha;•u(n) + &epsilon;(n)**, where **&alpha;** is a constant and **epsilon;(n)** is a number generated from a random normal distribution

*Example 2:* a recurrent neural network (RNN) is a discrete timeseries defined by **u(n+1) = u(n) + f(u(n), &theta;)** where **f** is a neural network parametrization and **&theta;** is the set of model parameters

### Properties of Linear Dynamical Systems
T
