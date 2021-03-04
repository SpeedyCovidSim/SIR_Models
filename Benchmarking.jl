#=
This is a code base for benchmarking a Python and Julia instance of a
for a simple SIR simulation using the Gillespie Direct Method

Author: Joel Trent and Josh Looker

For info on Benchmark:
https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/doc/manual.md#handling-benchmark-results
=#

#Pkg.add("BenchmarkTools")
#Pkg.add("PyCall")
#Pkg.build("Pycall") #in case Julia doesn't manage to build the package
#Pkg.add("Conda")

using Conda
using BenchmarkTools
using PyCall

# Julia version of sumArray
function jlSumArray(a)
    arraySum = 0.0
    for i in 1:length(a)
        arraySum += a[i]
    end
    return sum
end


# Create a reasonably large array to compute the sum of
a = rand(10^7)

# Py looks in current directory, may be unneccesary
pushfirst!(PyVector(pyimport("sys")."path"), "")

# py code break to import the function desired
py"""
from BenchmarkTest import sumArray
"""

# make locally accessible for benchmark
pySumArray = py"sumArray"

# using the function
pySumArray(a)

pySumMark = @benchmark pySumArray(a)

jlSumMark = @benchmark jlSumArray(a)

numpySum = pyimport("numpy")."sum"

pyNumpySumMark = @benchmark numpySum(a)
