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
#Pkg.add("Measures")

using Conda, BenchmarkTools, PyCall, Plots, Measures

# It'd probably be better to tell Julia to use an already installed python ENV
#Conda.add("matplotlib")

#-----------------------------------------------------------------------------

function plots(tMean,tMedian, N, Display=true, save=true)
    #=
    Inputs
    tMean   : 2D array. Col 1 contains mean times for Julia function, Col 2 for
              Python to complete simulation.
    tMedian : 2D array. Col 1 contains median times for Julia function, Col 2 for
              Python to complete simulation.
    N       : Array of Population size used.

    Outputs
    png     : plot of SIR model over time [by default]
    =#

    gr(reuse=true)

    # MAY NEED TO USE EXP or log time if order of magnitude between times taken
    # need to use log N

    # Use margin to give white space around titling
    plot1 = plot(log10.(N), log10.(tMean), label=["Julia" "Python"], lw = 2, margin = 5mm)
    plot2 = plot(log10.(N), log10.(tMedian), label=["Julia" "Python"], lw = 2, margin = 5mm)
    plot(plot1, plot2, layout = (1, 2), title = ["Mean time to complete the Simulation" "Median time to complete the Simulation"])
    #plot!(t, I, label="Infected", show = true)
    #display(plot!(t, R, label="Recovered", show = true))

    plot!(size=(1000,500))
    xlabel!("Log10 Population size used")
    ylabel!("Log10 Simulation time (log10(s))")

    if Display
        # required to display graph on plots.
        display(plot!())
    end

    if save
        # Save graph as pngW
        png("SimulationTimes")
    end
end


# Benchmarking Python vs Julia -----------------------------------------------

# Loading a Julia function (must be within a module). Set path to current
# CHANGE the module WHEN 'MAIN' BRANCH UPDATED
push!( LOAD_PATH, "./" )
using moduleTest # found this better than doing 'include'

# create local function wrapper
juliaGillespieDirect = moduleTest.gillespieDirect2Processes

# Py looks in current directory, required
pushfirst!(PyVector(pyimport("sys")."path"), "")

# import python function. Uncomment when ready
py"""
from GillespieDirect import gillespieDirect2Processes
"""

# make local function wrapper for benchmark
pythonGillespieDirect = py"gillespieDirect2Processes"

# Setup the inputs for the test
N = [5, 10, 50, 100,1000,10000,100000]

S_total = N .- 1
I_total = N .* 0 .+ 1
R_total = N .* 0

t_max = 200
alpha = 0.4
beta = 0.0004

tMean = zeros(Float64,length(N),2)
tMedian = zeros(Float64,length(N),2)

# setup a loop that benchmarks each function for different N values
for i in 1:length(N)

    jlGillespieTimes = []
    pyGillespieTimes = []

    for j in 1:(100/log10(N[i]))
        push!(jlGillespieTimes, @elapsed juliaGillespieDirect(t_max, S_total[i], I_total[i], R_total[i], alpha, beta, N[i]))
        push!(pyGillespieTimes, @elapsed pythonGillespieDirect(t_max, S_total[i], I_total[i], R_total[i], alpha, beta, N[i]))
    end

    println("Completed iteration #$i")

    tMean[i,:] = [mean(jlGillespieTimes), mean(pyGillespieTimes)]
    tMedian[i,:] = [median(jlGillespieTimes), median(pyGillespieTimes)]
end

#println(tMean)

# graph the benchmark of time as N increases. Recommend two graphs next to
# each other with median on one and mean on other.
plots(tMean,tMedian,N,true,false)


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
