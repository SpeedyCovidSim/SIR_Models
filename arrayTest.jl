# testing speed of accessing col vs rows

using Distributions, Random, BenchmarkTools




n = 10000000

x = cumsum(ones(n,5), dims=2)

x2 = copy(rotr90(x))


@benchmark rand.(Exponential.(x))

@benchmark rand.(Exponential.(x2))


n=10
a = convert.(Int,cumsum(ones(n*2)))

b = zeros(2,n)


b = divrem.(a .-1, 10) 

I1 = [[x[1],x[2]] for x in b]

b[:,2] .+= 1
