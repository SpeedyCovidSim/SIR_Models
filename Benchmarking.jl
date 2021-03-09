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


using Conda, BenchmarkTools, PyCall

# It'd probably be better to tell Julia to use an already installed python ENV

#-----------------------------------------------------------------------------

# Benchmarking Python vs Julia -----------------------------------------------

# Loading a Julia function (must be within a module). Set path to current

# import required modules
push!( LOAD_PATH, "./" )
using sirModels: gillespieDirect2Processes_rand # don't use 'include'
using plots: plotBenchmarks

# create local function wrapper
juliaGillespieDirect = gillespieDirect2Processes_rand

# Py looks in current directory, required
pushfirst!(PyVector(pyimport("sys")."path"), "")

# import python function. Uncomment when ready
py"""
from sirModels import gillespieDirect2Processes
"""

# make local function wrapper for benchmark
pythonGillespieDirect = py"gillespieDirect2Processes"

# Setup the inputs for the test
N = [5, 10, 50,100,1000,10000,100000]

S_total = N .- 1
I_total = ceil.(0.05 .* N)
R_total = zeros(length(N))

t_max = 200
alpha = 0.4
beta = 10 ./ N

tMean = zeros(Float64,length(N),2)
tMedian = zeros(Float64,length(N),2)

# setup a loop that benchmarks each function for different N values
for i in 1:length(N)

    jlGillespieTimes = []
    pyGillespieTimes = []

    for j in 1:(400/log10(N[i]))
        push!(jlGillespieTimes, @elapsed juliaGillespieDirect(t_max, S_total[i], I_total[i], R_total[i], alpha, beta[i], N[i]))
        push!(pyGillespieTimes, @elapsed pythonGillespieDirect(t_max, S_total[i], I_total[i], R_total[i], alpha, beta[i], N[i]))
    end

    println("Completed iteration #$i")

    tMean[i,:] = [mean(jlGillespieTimes), mean(pyGillespieTimes)]
    tMedian[i,:] = [median(jlGillespieTimes), median(pyGillespieTimes)]

end

println(tMean)

meanSpeedup = tMean[:,2]./tMean[:,1]
medianSpeedup = tMedian[:,2]./tMedian[:,1]

println("Mean Speedup of: $meanSpeedup")
println("Median Speedup of: $medianSpeedup")

# graph the benchmark of time as N increases. Recommend two graphs next to
# each other with median on one and mean on other.
plotBenchmarks(tMean,tMedian,N,true,false)


# Test benchmark on summing an array -----------------------------------------
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
