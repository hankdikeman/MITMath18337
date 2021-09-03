# Newton iteration method for finding the quantile of an arbitrary continuous distribution
cd(@__DIR__)
using Pkg
Pkg.activate("./H1Environment")

using Distributions

# newton iteration to find quantile of distribution
function newton_quantile(d::UnivariateDistribution, y::Float64, x0::Float64=mean(d), tol::Float64=1E-4)::Float64
    # iterate while error above tolerance
    while abs(cdf(d, x0) - y) > tol
        x0 = x0 - (cdf(d, x0) - y) / (pdf(d, x0))
    end
    x0
end

using BenchmarkTools

g = Gamma(5, 1) 
n = Normal(0, 1)
b = Beta(2, 4)
newton_quantile(g,0.99)
# test on gamma function
@btime quantile(g, 0.78)
@btime newton_quantile(g, 0.78)
# test on normal function
@btime quantile(n, 0.78)
@btime newton_quantile(n, 0.78)
# test on beta function
@btime quantile(b, 0.78)
@btime newton_quantile(b, 0.78)
