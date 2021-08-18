# testing speed of accessing col vs rows

using Distributions, Random, BenchmarkTools

n = 10000000

x = cumsum(ones(n,5), dims=2)

x2 = copy(rotr90(x))


@benchmark rand.(Exponential.(x))

@benchmark rand.(Exponential.(x2))


n=10
a = convert.(Int,cumsum(ones(n*2)))

b = convert.(Int, zeros(n*2,2))


b[:,1] = div.(a .-1, 10) .+1
b[:,2] = rem.(a .-1, 10) .+1

println(b)
