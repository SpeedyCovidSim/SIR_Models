#=
Test the functionality of creating Python function wrappers, and using Benchmark
on them.
=#
using Conda, BenchmarkTools, PyCall
# push!( LOAD_PATH, "./" )

# Julia version of sumArray
function jlSumArray(a)
    arraySum = 0.0
    for i::Int64 in 1:length(a)
        @inbounds arraySum += a[i]
    end
    return sum
end

# Create a reasonably large array to compute the sum of
a = rand(10^7)

# Py looks in current directory
pushfirst!(PyVector(pyimport("sys")."path"), "./JuliaTestFiles/")

# py code break to import the function desired
py"""
from BenchmarkTest import sumArray
"""

# make local function wrapper for benchmark
pySumArray = py"sumArray"

# using the function
pySumArray(a)

pySumMark = @benchmark pySumArray(a)

jlSumMark = @benchmark jlSumArray(a)

numpySum = pyimport("numpy")."sum"

pyNumpySumMark = @benchmark numpySum(a)

# println(pySumMark)
#
# println(jlSumMark)
#
# println(pyNumpySumMark)
