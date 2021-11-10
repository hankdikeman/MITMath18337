# MITMath18337

This repo contains the programs developed for the parallel computing/scientific ML class 18.337 on [MIT OpenCourseware](https://mitmath.github.io/18337/)

1. Lecture 1
   - [Lecture 1 Notes](Lecture1/Notes_Lecture1/): contains typed notes on intro to course, intro to Julia, and some Julia resources
   - [Lecture 1 Programs](Lecture1/Programs_Lecture1/): contains some simple Julia stuff, my first Julia program, an environment with some autodiff tools
2. Lecture 2
   - [Lecture 2 Notes](Lecture2/Notes_Lecture2/): contains notes on optimizing serial code, with a bit of a special emphasis in optimizing compiled Julia programs
   - [Lecture 2 Programs](Lecture2/Programs_Lecture2/): some optimization work for several ODE systems, a fully optimized (cached variables, inplace ops, inference engine optimization) PDE system implementation
3. Lecture 3
   - [Lecture 3 Notes](Lecture3/Notes_Lecture3/): typed notes on an intro to scientific machine learning, with a quick intro to `Flux.jl` and details on a common scientific ML framework
   - [Lecture 3 Programs](Lecture3/Programs_Lecture3/): some simple implementations of trivial neural networks, and a PINN for a damped oscillator system
4. Lecture 4
   - [Lecture 4 Notes](Lecture4/Notes_Lecture4/): typed notes on discrete dynamical systems and efficient implementations to solve for these systems in Julia and compiled languages in general
   - [Lecture 4 Programs](Lecture4/Programs_Lecture4/): an implementation to solve for the discretized form of the Lorenz system, with naive and optimized implementations
5. Lecture 5
   - [Lecture 5 Notes](Lecture5/Notes_Lecture5/): typed notes on parallelism, including array-based, memory parallelism, and "embarrassingly parallel" problems
   - [Lecture 5 Programs](Lecture5/Programs_Lecture5/): an implementation of multithreaded Lorenz dynamics and a multithreaded parameter search (i.e., embarrassingly parallel problems)
6. Lecture 6
   - [Lecture 6 Notes](Lecture6/Notes_Lecture6/): typed notes on types of parallelism, including graph-based, explicit and implicit array-based, GPU parallelism, and map-reduce parallelism
   - [Lecture 6 Programs](Lecture6/Programs_Lecture6/): implementations of each of the above types of parallelism (though GPU is not possible and MPI parallelism is still a WIP)
7. Lecture 7
   - [Lecture 7 Notes](Lecture7/Notes_Lecture7/): typed notes on an overview to ODEs and their applications in scientific modeling, as well as an overview of some of the challenges of working with ODE systems like stiffness and stability
   - [Lecture 7 Programs](Lecture7/Programs_Lecture7/): none yet
8. Lecture 8
   - [Lecture 8 Notes](Lecture8/Notes_Lecture8/): typed notes on the basic operation of forward-mode autodifferentiation, as well as the data structures and algebraic rules underlying autodiff in general
   - [Lecture 8 Programs](Lecture8/Programs_Lecture8/): an implementation of a Newton Fixed-point Solver using the Julia package `ForwardDiff.jl`
9. Lecture 9
   - [Lecture 9 Notes](Lecture9/Notes_Lecture9/): typed notes on the different implicit methods used to solve for ODE systems, especially stiff ODE systems, and efficiently computing Jacobian matrices using automatic differentiation
   - [Lecture 9 Programs](Lecture9/Programs_Lecture9/): none yet
