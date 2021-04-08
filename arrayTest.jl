# testing speed of accessing col vs rows

using Distributions, Random, BenchmarkTools




n = 10000000

x = cumsum(ones(n,5), dims=2)

x2 = copy(rotr90(x))


@benchmark rand.(Exponential.(x))

@benchmark rand.(Exponential.(x2))
