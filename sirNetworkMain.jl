#=
This is a code base for the main function of a network SIR simulation using the
Gillespie Direct Method

Author: Joel Trent and Josh Looker
=#
using Random, Conda, PyCall, LightGraphs, GraphPlot, MetaGraphs


# import required modules
push!( LOAD_PATH, "./" )    #Set path to current
using networkFunctions: initialiseNetwork!
using sirModels: gillespieDirect_network!
using plotsPyPlot: plotSIRPyPlot

# main
function main(Display = true, save = true)

    # testing the gillespieDirect2Processes_network functions

    # Get same thing each time
    Random.seed!(1)

    # initialise variables
    N = [5,10,50,100,1000,10000]

    t_max = 200
    alpha = 0.4
    beta = 10 ./ N

    # new inputs to the initialisation
    infectionProp = 0.05
    states = ["S","I","R"]
    stateEvents = [["I"],["R"],nothing]
    eventHazards = [[beta], [alpha], [0]]
    HazardMultipliers = [["I"],nothing,nothing]


    if false
        # iterate through populations. Complete Graph
        for i in 1:length(N)

            println("Iteration #$i commencing")
            # initialise the network
            network = MetaGraph(complete_graph(N[i]))
            println("Network #$i returned")

            initialiseNetwork!(network, infectionProp, states, stateEvents, eventHazards)

            println("Network #$i has been initialised")

            time = @elapsed t, state_Totals = gillespieDirect_network!(t_max, network, alpha, beta[i], N[i])

            S = state_Totals[1,:]
            I = state_Totals[2,:]
            R = state_Totals[3,:]

            println("Simulation #$i has completed in $time")

            if Display | save
                population = N[i]
                outputFileName = "juliaGraphs/networkCompleteDirect/SIR_Model_Pop_$population"
                plotSIRPyPlot(t, [S,I,R], alpha, beta[i], N[i], outputFileName, Display, save)
            end
        end
    end


    # iterate through populations
    for i in 1:length(N)

        println("Iteration #$i commencing")
        # initialise the network
        k = [2,3,10,20,100,1000] # degree of connection

        network = MetaGraph(random_regular_graph(N[i], k[i]))
        println("Network #$i returned")

        initialiseNetwork!(network, infectionProp, states, stateEvents, eventHazards)

        println("Network #$i has been initialised")

        time = @elapsed t, state_Totals = gillespieDirect_network!(t_max, network, alpha, beta[i], N[i])

        S = state_Totals[1,:]
        I = state_Totals[2,:]
        R = state_Totals[3,:]

        println("Simulation #$i has completed in $time")

        if Display | save
            population = N[i]
            outputFileName = "juliaGraphs/networkSmallDegreeDirect/SIR_Model_Pop_$population"
            plotSIRPyPlot(t, [S,I,R], alpha, beta[i], N[i], outputFileName, Display, save)
        end
    end


end

# main(Display, save)
main(true, false)
