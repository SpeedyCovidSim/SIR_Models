#=
This is a code base for the main function of a network SIR simulation using the
Gillespie Direct Method

Author: Joel Trent and Josh Looker
=#
using Random, Conda, PyCall, LightGraphs, GraphPlot#, MetaGraphs


# import required modules
push!( LOAD_PATH, "./" )    #Set path to current
using networkInitFunctions: initialiseNetwork!
#using sirModels: gillespieDirect_network!, gillespieFirstReact_network!
using plotsPyPlot: plotSIRPyPlot

# main
function main(direct, firstReact, nextReact, Display = true, save = true, profiling = true)

    # testing the gillespieDirect_network function
    if direct
        # Get same thing each time
        Random.seed!(1)

        # initialise variables
        N = [5,10,50,100,1000,10000]

        t_max = 200
        alpha = 0.4
        beta = 0 ./ N .+ 10
        gamma = 0.1

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIRD_direct"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,100,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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
                outputFileName = "juliaGraphs/networkCompleteDirect/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,100,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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
                outputFileName = "juliaGraphs/networkSmallDegreeDirect/SIR_Model_Pop_$population"
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
        alpha = 0.4
        beta = 0 ./ N .+ 10
        gamma = 0.1

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIRD_firstReact"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,100,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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
                outputFileName = "juliaGraphs/networkCompleteFirstReact/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,100,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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
                outputFileName = "juliaGraphs/networkSmallDegreeFirstReact/SIR_Model_Pop_$population"
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
        alpha = 0.4
        beta = 0 ./ N .+ 10
        gamma = 0.1

        # new inputs to the initialisation
        infectionProp = 0.05
        simType = "SIRD_nextReact"


        if profiling

            # profiling the gillespieDirect_network function
            for i in length(N):length(N)
                println("Iteration #$i commencing")
                # initialise the network
                k = [2,3,10,20,100,1000] # degree of connection

                network = random_regular_graph(N[i], k[i])
                println("Network #$i returned")

                networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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
                outputFileName = "juliaGraphs/networkCompleteNextReact/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end

        # iterate through populations
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            k = [2,3,10,20,100,1000] # degree of connection

            network = random_regular_graph(N[i], k[i])
            println("Network #$i returned")

            networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(network, infectionProp, simType, alpha, beta[i], gamma)

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
                outputFileName = "juliaGraphs/networkSmallDegreeNextReact/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R,D], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end
    end
end

# main(direct, firstReact, nextReact, Display, save, profiling)
main(false, false, true, true, false, false)
