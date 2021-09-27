#=
This is a code base for the main function of a network SIR simulation using the
Gillespie Direct Method

Author: Joel Trent and Josh Looker
=#
using Random, Conda, PyCall, LightGraphs, GraphPlot#, MetaGraphs
using BenchmarkTools, Seaborn, DataFrames
using ProgressMeter

# import required modules
push!( LOAD_PATH, "./" )    #Set path to current
using networkInitFunctions: initialiseNetwork!
#using sirModels: gillespieDirect_network!, gillespieFirstReact_network!
using plotsPyPlot: plotSIRPyPlot, plotBenchmarks_network

global const PROGRESS__METER__DT = 0.2

function mainSIR(direct, firstReact, nextReact, Display = true, save = true, profiling = true)

    # testing the gillespieDirect_network function
    if direct
        # Get same thing each time
        Random.seed!(1)

        # initialise variables
        N = [5,10,50,100,1000,10000]

        t_max = 200
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIR_direct"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,200,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i], infectionProp, simType, alpha, beta[i], gamma)

                println("Network #$i has been initialised")

                @profiler model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            end
            #
        end

        # iterate through populations. Complete Graph
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            network = complete_graph(N[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i], infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            # D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkCompleteDirect/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,200,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i], infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            #t, state_Totals = gillespieDirect_network!(t_max, network, alpha, beta[i], N[i])

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            # D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkSmallDegreeDirect/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end
    end

    # testing the gillespieFirstReact_network function
    if firstReact
        # Get same thing each time
        Random.seed!(1)

        # initialise variables
        N = [5,10,50,100,1000,10000]

        t_max = 200
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIR_firstReact"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,200,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

                println("Network #$i has been initialised")

                @profiler model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            end
            #
        end

        # iterate through populations. Complete Graph
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            network = complete_graph(N[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            # D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkCompleteFirstReact/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,200,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            #t, state_Totals = gillespieDirect_network!(t_max, network, alpha, beta[i], N[i])

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            # D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkSmallDegreeFirstReact/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end
    end

    # testing the gillespieFirstReact_network function
    if nextReact
        # Get same thing each time
        Random.seed!(1)

        # initialise variables
        N = [5,10,50,100,1000,10000]

        t_max = 200
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIR_nextReact"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,200,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

                println("Network #$i has been initialised")

                @profiler model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            end
            #
        end

        # iterate through populations. Complete Graph
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            network = complete_graph(N[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            # D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkCompleteNextReact/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,200,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            #t, state_Totals = gillespieDirect_network!(t_max, network, alpha, beta[i], N[i])

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            # D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkSmallDegreeNextReact/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end
    end
end

# main
function mainSIRD(direct, firstReact, nextReact, Display = true, save = true, profiling = true)

    # testing the gillespieDirect_network function
    if direct
        # Get same thing each time
        Random.seed!(1)

        # initialise variables
        N = [5,10,50,100,1000,10000]

        t_max = 200
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIRD_direct"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,200,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i], infectionProp, simType, alpha, beta[i], gamma)

                println("Network #$i has been initialised")

                @profiler model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            end
            #
        end

        # iterate through populations. Complete Graph
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            network = complete_graph(N[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i], infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkCompleteDirect/SIRD_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,200,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i], infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            #t, state_Totals = gillespieDirect_network!(t_max, network, alpha, beta[i], N[i])

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkSmallDegreeDirect/SIRD_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end
    end

    # testing the gillespieFirstReact_network function
    if firstReact
        # Get same thing each time
        Random.seed!(1)

        # initialise variables
        N = [5,10,50,100,1000,10000]

        t_max = 200
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIRD_firstReact"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,200,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

                println("Network #$i has been initialised")

                @profiler model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            end
            #
        end

        # iterate through populations. Complete Graph
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            network = complete_graph(N[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkCompleteFirstReact/SIRD_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,200,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            #t, state_Totals = gillespieDirect_network!(t_max, network, alpha, beta[i], N[i])

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkSmallDegreeFirstReact/SIRD_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end
    end

    # testing the gillespieFirstReact_network function
    if nextReact
        # Get same thing each time
        Random.seed!(1)

        # initialise variables
        N = [5,10,50,100,1000,10000]

        t_max = 200
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIRD_nextReact"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,200,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

                println("Network #$i has been initialised")

                @profiler model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            end
            #
        end

        # iterate through populations. Complete Graph
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            network = complete_graph(N[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkCompleteNextReact/SIRD_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,200,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
            #t, state_Totals = gillespieDirect_network!(t_max, network, alpha, beta[i], N[i])

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]
            D = state_Totals[4,:]
            # D = []

            lengthStateTotals = length(state_Totals)
            println("Length of state_Totals array is $lengthStateTotals")

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkSmallDegreeNextReact/SIRD_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end
    end
end

function main(SIR, SIRD)

    mainSIR(true, true, true, false, true, false)

    # main(direct, firstReact, nextReact, Display, save, profiling)
    mainSIRD(true, true, true, false, true, false)
    return nothing
end

# main(true, true)

function benchmarkNetwork(compare3, compare2, violin3)

    if compare3
        # initialise variables
        N = [50,100,500,1000,10000,30000]
        k = [30,50,200,500,1000,2500] # degree of connection

        it_count = [400, 200, 20, 5, 3, 2] .*2
        # 1000 ./ (log10.([5,10,50,100,1000,10000]).^3) .* [1,1,1,1,1,0.1]

        t_max = 40
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05


        tMean = zeros(Float64,length(N),3)
        tMedian = zeros(Float64,length(N),3)

        # setup a loop that benchmarks each function for different N values
        for i::Int64 in 1:length(N)

            directTimes = []
            firstTimes = []
            nextTimes = []

            for j in 1:(it_count[i])
                network = random_regular_graph(N[i], k[i])

                simType = "SIR_direct"
                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                push!(directTimes, @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState))

                simType = "SIR_firstReact"
                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                push!(firstTimes, @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState))

                simType = "SIR_nextReact"
                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                push!(nextTimes, @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState))
            end


            println("Completed iteration #$i")

            tMean[i,:] = [mean(directTimes), mean(firstTimes), mean(nextTimes)]
            tMedian[i,:] = [median(directTimes), median(firstTimes), median(nextTimes)]

        end

        println(tMean)

        # graph the benchmark of time as N increases. Recommend two graphs next to
        # each other with median on one and mean on other.
        plotBenchmarks_network(tMean,tMedian,N,true,true)
    end
    ##########################

    if compare2

        # initialise variables
        N = [50,100,500,1000,10000,30000]
        k = [30,50,200,500,1000,2500] # degree of connection

        it_count = [400, 200, 20, 5, 3, 2] .* 2
        # 1000 ./ (log10.([5,10,50,100,1000,10000]).^3) .* [1,1,1,1,1,0.1]

        t_max = 40
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05


        tMean = zeros(Float64,length(N),2)
        tMedian = zeros(Float64,length(N),2)

        # setup a loop that benchmarks each function for different N values
        for i::Int64 in 1:length(N)

            directTimes = []
            nextTimes = []

            for j in 1:(it_count[i])
                network = random_regular_graph(N[i], k[i])

                simType = "SIR_direct"
                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                push!(directTimes, @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState))

                simType = "SIR_nextReact"
                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                push!(nextTimes, @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState))
            end


            println("Completed iteration #$i")

            tMean[i,:] = [mean(directTimes), mean(nextTimes)]
            tMedian[i,:] = [median(directTimes), median(nextTimes)]

        end

        println(tMean)

        # graph the benchmark of time as N increases. Recommend two graphs next to
        # each other with median on one and mean on other.
        outputFileName = "Benchmarks/SimulationTimesNetwork_DiscVNext"
        plotBenchmarks_network(tMean,tMedian,N,["Discrete" "Next React"], outputFileName, true,true)

    end

    if violin3
        # initialise variables
        N = [20,50,100,500,1000,10000, 20000]
        k = [ 4,10, 20,100, 200, 1000,  2000] # degree of connection

        it_count = [1000, 800, 400, 100, 50, 10, 5] .*4
        it_count_first = [1000, 800, 200, 50, 20, 4, 2]
        # 1000 ./ (log10.([5,10,50,100,1000,10000]).^3) .* [1,1,1,1,1,0.1]

        t_max = 40
        alpha = 0.15
        beta = 0 ./ N .+ 1.5
        gamma = alpha * 0.25

        # new inputs to the initialisation
        infectionProp = 0.05

        tMean = zeros(Float64,length(N),3)
        tMedian = zeros(Float64,length(N),3)
        time_df = DataFrame(time=Float64[], type=String[], population=Int64[])

        # setup a loop that benchmarks each function for different N values
        for i::Int64 in 1:length(N)

            directTimes = []
            firstTimes = []
            nextTimes = []

            # Random.seed!(j)
            networks = [random_regular_graph(N[i], k[i]) for _ in 1:(it_count[i])]

            for j in 1:(it_count[i])
                Random.seed!(j)
                # network = random_regular_graph(N[i], k[i])
                network = networks[j]

                simType = "SIR_direct"
                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                time_dir = @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
                push!(directTimes, time_dir)
                push!(time_df, [log10(time_dir), "Direct", N[i]])

                # simType = "SIR_firstReact"
                # networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                # time_fir = @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
                # push!(firstTimes, time_fir)
                # push!(time_df, [log10(time_fir), "First Reaction", N[i]])

                # Random.seed!(j)
                # simType = "SIR_nextReact"
                # networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                # time_nex = @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
                # push!(nextTimes, time_nex)
                # push!(time_df, [log10(time_nex), "Next Reaction", N[i]])
            end

            for j in 1:(it_count_first[i])
                Random.seed!(j)
                # network = random_regular_graph(N[i], k[i])
                network = networks[j]

                simType = "SIR_firstReact"
                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                time_fir = @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
                push!(firstTimes, time_fir)
                push!(time_df, [log10(time_fir), "First Reaction", N[i]])
            end

            for j in 1:(it_count[i])
                Random.seed!(j)
                # network = random_regular_graph(N[i], k[i])
                network = networks[j]

                simType = "SIR_nextReact"
                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, N[i],infectionProp, simType, alpha, beta[i], gamma)
                time_nex = @elapsed model!(t_max, network, alpha, beta[i], N[i], networkVertex_df, network_dict, stateTotals, isState)
                push!(nextTimes, time_nex)
                push!(time_df, [log10(time_nex), "Next Reaction", N[i]])
            end

            println("Completed iteration #$i")

            tMean[i,:] = [mean(directTimes), mean(firstTimes), mean(nextTimes)]
            tMedian[i,:] = [median(directTimes), median(firstTimes), median(nextTimes)]

        end

        println(tMean)
        println(tMedian)
        # graph the benchmark of time as N increases. Recommend two graphs next to
        # each other with median on one and mean on other.
        outputFileName = "Benchmarks/SimulationTimesNetwork_DiscVNext"
        plotBenchmarks_network(tMean,tMedian,N,["Discrete" "First React" "Next React"], outputFileName, true,true)

        Seaborn.set()
        Seaborn.set_style("ticks")
        fig = plt.figure(dpi=300)
        # plt.violinplot(data)
        Seaborn.violinplot(x=time_df.population, y=time_df.time, hue=time_df.type,
            bw=1.5, cut=0, width = 0.9, scale="width", palette = "Set2", aspect=.7, gridsize=40, saturation=0.9)

        plt.xlabel("Population Size")
        plt.ylabel("Log10 Simulation time (log10(s))")
        plt.title("Time To Complete Simulation")
        # plt.title("For alpha = $alpha and beta $beta")
        plt.legend(loc = "upper left")
        display(fig)
        fig.savefig("Benchmarks/SimulationTimesNetwork")
        close()

    end
end

benchmarkNetwork(false, false, true)
