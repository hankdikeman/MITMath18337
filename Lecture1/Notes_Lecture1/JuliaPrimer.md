## Intro to Julia Programming Language
08/08/21

This class uses the Julia programming language, since it easier to maintain programming environments, Julia has a lot of community support, but most importantly it is possible to dig into the assembly and low level code during optimization and profiling of programs (serial and parallel)


I installed Julia with homebrew, using command `brew install --cask julia`. One complication for my Julia installation is that I had trouble setting the number of threads present on my CPU. Normally one would export the environment variable `JULIA_NUM_THREADS=__` but this didn't seem to work. Instead, one can call Julia using the command line option `threads`, as so: `julia --threads=8`. If I don't do this, none of the multithreading work I do will accomplish anything

### Installing and loading packages
- Use `]` inside REPL to enter package manager
- `activate _____` to activate environment
- `add _____` to install and load packages
- `remove _____` to remove loaded packages


Julia environments are automatically created inside a directory with the name of the environment, containing 2 files which detail the environment and dependencies included. (`.toml` files). I need to read more to understand how package management differs from Python. Lots of package info can be found on [JuliaHub](https://juliahub.com/lp/)


### Package generally includes
- package dependencies
- `include` commands to bring in subfiles
- exports of needed functions and variables
- automatic tests


One thing I need to sort out before I move forward is the syntax highlighting in vim for Julia. It'll be the death of me writing programs with no syntax highlighting, I need that fixed pronto.


Should honestly spend 2-3 hours picking through resources so I set good habits before I try any of the homework
