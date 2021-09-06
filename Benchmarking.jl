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

using Conda, BenchmarkTools, PyCall, DataFrames, Seaborn
using ProgressMeter

# It'd probably be better to tell Julia to use an already installed python ENV

# ##############################################################################

# Benchmarking Python vs Julia #################################################

# Loading a Julia function (must be within a module). Set path to current

# import required modules
push!( LOAD_PATH, "./" )
using sirModels: gillespieDirect2Processes_rand # don't use 'include'
using plotsPyPlot: plotBenchmarksViolin


function gillespieDirect_pyVsJl()
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
    N = [5, 25, 50,100,1000,10000,100000]

    S_total = N - ceil.(0.05 .* N)
    I_total = ceil.(0.05 .* N)
    R_total = zeros(length(N))

    t_max = 200
    alpha = 0.4
    beta = 10 ./ N

    # @profiler juliaGillespieDirect(t_max, S_total[end], I_total[end], R_total[end], alpha, beta[end], N[end])

    tMean = zeros(Float64,length(N),2)
    tMedian = zeros(Float64,length(N),2)

    time_df = DataFrame(time=Float64[], language=String[], population=Int64[])

    # push!(a, [1.0, "Python", 10])
    # push!(a, [2.0, "Julia", 10])

    # setup a loop that benchmarks each function for different N values
    for i::Int64 in 1:length(N)

        jlGillespieTimes = []
        pyGillespieTimes = []

        @showprogress 0.5 for j in 1:(20000/log10(N[i]))
            time_jl = @elapsed juliaGillespieDirect(t_max, copy(S_total[i]), copy(I_total[i]), copy(R_total[i]), copy(alpha), copy(beta[i]), copy(N[i]))
            push!(jlGillespieTimes, time_jl)
            push!(time_df, [log10(time_jl), "Julia", N[i]])

            time_py =  @elapsed pythonGillespieDirect(t_max, copy(S_total[i]), copy(I_total[i]), copy(R_total[i]), copy(alpha), copy(beta[i]), copy(N[i]))
            push!(pyGillespieTimes, time_py)
            push!(time_df, [log10(time_py), "Python", N[i]])
        end

        println(S_total[i])

        println("Completed iteration #$i")

        # tMean[i,:] = [mean(jlGillespieTimes), mean(pyGillespieTimes)]
        # tMedian[i,:] = [median(jlGillespieTimes), median(pyGillespieTimes)]

    end

    # println(tMean)
    #
    # meanSpeedup = tMean[:,2]./tMean[:,1]
    # medianSpeedup = tMedian[:,2]./tMedian[:,1]
    #
    # println("Mean Speedup of: $meanSpeedup")
    # println("Median Speedup of: $medianSpeedup")
    #
    # # graph the benchmark of time as N increases. Recommend two graphs next to
    # # each other with median on one and mean on other.
    # plotBenchmarks(tMean,tMedian,N,true,true)

    outputFileName = "Benchmarks/SimulationTimes"
    xlabel = "Population Size"
    plotBenchmarksViolin(time_df.population, time_df.time, time_df.language, outputFileName,
        xlabel, true, true)

    # Seaborn.set()
    # # Seaborn.set_style("ticks")
    # fig = plt.figure(dpi=300)
    # # plt.violinplot(data)
    # Seaborn.violinplot(x=time_df.population, y=time_df.time, hue=time_df.language,
    #     bw=0.5, cut=0,scale="count",palette = "Set2" )
    #
    # plt.xlabel("Population Size")
    # plt.ylabel("Log10 Simulation time (log10(s))")
    # plt.title("Time To Complete Simulation")
    # # plt.title("For alpha = $alpha and beta $beta")
    # plt.legend()
    # display(fig)
    # fig.savefig("Benchmarks/SimulationTimes")
    # close()
end

gillespieDirect_pyVsJl()
