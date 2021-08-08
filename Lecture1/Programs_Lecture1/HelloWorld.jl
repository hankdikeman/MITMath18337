cd(@__DIR__)
using Pkg
Pkg.activate("./L1Environment")

using ForwardDiff, FiniteDiff

f(x) = 2x^2 + x
println(ForwardDiff.derivative(f,2))
println(FiniteDiff.finite_difference_derivative(f, 2.0))

using PkgTemplates
t = Template(user="hankdikeman")
t("L1Environment_Temp")
