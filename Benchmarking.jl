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


# Benchmarking Python vs Julia
if true

    # Loading a Julia function (must be within a module). Set path to current
    # CHANGE WHEN 'MAIN' BRANCH UPDATED
    push!( LOAD_PATH, "./" )
    using moduleTest # found this better than doing 'include'

    # create local function wrapper
    juliaGillespieDirect = moduleTest.gillespieDirect2Processes

    # import python function. Uncomment when ready
    #py"""
    #from ???? import ???
    #"""

    # make local function wrapper for benchmark
    # pythonGillespieDirect = py"???"

    # Setup the inputs for the test
    N = [5, 10, 50, 100,1000,10000,100000]

    S_total = N .- 1
    I_total = N .* 0 .+ 1
    R_total = N .* 0

    t_max = 200
    alpha = 0.4
    gamma = 0.0004

    i = 7

    # setup a loop that benchmarks each function for different N values

    # pyGillespieMark = @benchmark pythonGillespieDirect(t_max, S_total[i], I_total[i], R_total[i], alpha, gamma, N[i])

    jlGillespieMark = @benchmark juliaGillespieDirect(t_max, S_total[i], I_total[i], R_total[i], alpha, gamma, N[i])

    # graph the benchmark of time as N increases. Recommend two graphs next to
    # each other with median on one and mean on other.

end



# Test benchmark on summing an array
if false

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

    # make local function wrapper for benchmark
    pySumArray = py"sumArray"

    # using the function
    pySumArray(a)

    pySumMark = @benchmark pySumArray(a)

    jlSumMark = @benchmark jlSumArray(a)

    numpySum = pyimport("numpy")."sum"

    pyNumpySumMark = @benchmark numpySum(a)

end
