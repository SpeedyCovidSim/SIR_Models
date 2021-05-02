module networkInitFunctions

    using LightGraphs, StatsBase, DataFrames#, MetaGraphs
    using sirModels: gillespieDirect_network!, gillespieFirstReact_network!,
        gillespieNextReact_network!
    using networkFunctions: incrementInfectedNeighbors!

    export initialiseNetwork!

    function initialiseNetwork!(network, infectionProp, simType, alpha, beta, gamma = 0)
        #=
        Inputs
        network       : a undirected graph from LightGraphs
                        containing our population of interest
        infectionProp : proportion of people to be infected [0, 1)
        simType       : type of sim to use. ["SIR", "SIRD"]

        Outputs
        network       : works in place on the network. network is initialised
        S_total       : Num people susceptible to infection
        I_total       : Num people infected
        R_total       : Num people recovered
        =#

        states, stateEvents, eventHazards, hazardMultipliers, simpleHazards, model!,
            eventsPerState = simType!(simType, alpha, beta, gamma)

        # return num vertices
        numVertices = nv(network)

        # initialise dictionaries and arrays for storing network attributes.
        # THIS NEEDS TO BE CHANGED FOR SPEED LATER ON. DO NOT DECLARE TYPE "ANY"
        network_dict = Dict{String, Any}()
        stateTotals = convert.(Int, zeros(length(states)))

        # dataframe init -------------------------------------------------------
        # networkVertex_dict = Dict{Int64, Dict{String, Any}}()
        networkVertex_df, isState = networkVertexInit(network, numVertices, states)
        # ----------------------------------------------------------------------

        # randomly choose individuals to be infected
        infectedVertices = sample(1:numVertices, Int(ceil(infectionProp *
            numVertices)), replace = false)

        # Set them as infected and increment numInfectedNeighbors for their localNeighbourhood
        # by 1
        for i::Int64 in infectedVertices
            networkVertex_df[i, :state] = "I"
            isState[i,1] = false
            isState[i,2] = true

            incrementInfectedNeighbors!(network, networkVertex_df, i)
        end

        # count the num of vertices in each state
        for i in vertices(network)
            stateTotals += states .== networkVertex_df[i, :state]
            #get_prop(network, i, :state)
        end

        # add attributes to the network_dict
        network_dict = Dict("states"=>states, "population"=>numVertices,
            "simpleHazards"=>simpleHazards, "eventsPerState"=>eventsPerState,
            "allMultipliers"=>hazardMultipliers)

        stateIndex = 1
        for state in states

            network_dict[state] = Dict("stateIndex"=>stateIndex,
                "events"=>stateEvents[stateIndex], "eventHazards"=>eventHazards[stateIndex],
                "hazardMult"=>hazardMultipliers[stateIndex])

            stateIndex +=1
        end

        return networkVertex_df, network_dict, stateTotals, isState, model!
    end

    function networkVertexInit(network, numVertices, states)
        #=
        Initialise the vertices of a non-bipartite network.
        Returns a dataframe containing vertex attributes and a 2d boolean array
        with rows corresponding to individuals, columns corresponding to states
        and values true or false as to whether in a given state. Only one 'true'
        allowed per row.
        =#

        networkVertex_df = DataFrame()

        # dataframe columns init
        state = Array{String}(undef, numVertices)
        state .= states[1]

        # faster than converting from zeros
        connectivity = Array{Int64}(undef, numVertices)
        connectivity .= 0

        initInfectedNeighbors = Array{Int64}(undef, numVertices)
        initInfectedNeighbors .= 0

        # array containing whether or not the individual is "S"
        isState = convert.(Bool,zeros(numVertices, length(states)))
        isState[:,1] .= true

        # Initialise all states as "S", all numInfectedNeighbors as zero
        # states[1] will always be S/the initial state
        # for loop has i as the vertex index
        for i::Int64 in vertices(network)

            # use vertex index as the primary key for the dictionary
            # networkVertex_dict[i] = Dict("state"=>states[1], "initInfectedNeighbors"=>0)

            connectivity[i] = length(neighbors(network, i))
        end
        networkVertex_df.state = state
        networkVertex_df.initInfectedNeighbors = initInfectedNeighbors
        networkVertex_df.connectivity = connectivity

        return networkVertex_df, isState
    end

    function simType!(simType, alpha, beta, gamma)
        #=
        Inputs
        simType       : type of sim to use. ["SIR", "SIRD"]
        alpha         : rate of infected person recovering
        beta          : rate of susceptible person being infected
        gamma         : mortality rate of infected person

        Outputs
        states            : allowable simulation states
        stateEvents       : events that can happen to each state. E.g. "I" means
                            an individual in the state of the same index of states
                            as "I" is in stateEvents, means that individual can
                            become "I". i.e. for SIR, a "S" can become "I"
        eventHazards      : hazard of an event occurring. Where multiple events
                            can happen to a particular state, these are stored
                            in an array in order with the events
        hazardMultipliers : any relevant multiplier to use on the hazard - i.e.
                            dependence on the number of infected individuals
                            neighbouring the individual == "I". nothing otherwise


                            HAZARDMULTIPLIERS should always be considered to be
                            I/number of local neighbors - it's the proportion of
                            an infected neighbourhood
        =#

        if simType == "SIR_direct"
            states = ["S","I","R"]
            stateEvents = [["I"],["R"],[nothing]]
            eventHazards = [[beta], [alpha], [0.0]]
            hazardMultipliers = [["I"],[nothing],[nothing]]
            simpleHazards = true
            model! = gillespieDirect_network!
            eventsPerState = Int64[1,1,1]


        elseif simType == "SIRD_direct"
            states = ["S","I","R","D"]
            stateEvents = [["I"],["R","D"],[nothing],[nothing]]
            eventHazards = [[beta], [alpha, gamma], [0.0],[0.0]]
            hazardMultipliers = [["I"],[nothing, nothing],[nothing],[nothing]]
            simpleHazards = true
            model! = gillespieDirect_network!
            eventsPerState = Int64[1,2,1,1]

        elseif simType == "SIR_firstReact"
            states = ["S","I","R"]
            stateEvents = [["I"],["R"],[nothing]]
            eventHazards = [[beta], [alpha], [0.0]]
            hazardMultipliers = [["I"],[nothing],[nothing]]
            simpleHazards = true
            model! = gillespieFirstReact_network!
            eventsPerState = Int64[1,1,1]


        elseif simType == "SIRD_firstReact"
            states = ["S","I","R","D"]
            stateEvents = [["I"],["R","D"],[nothing],[nothing]]
            eventHazards = [[beta], [alpha, gamma], [0.0],[0.0]]
            hazardMultipliers = [["I"],[nothing, nothing],[nothing],[nothing]]
            simpleHazards = true
            model! = gillespieFirstReact_network!
            eventsPerState = Int64[1,2,1,1]

        elseif simType == "SIR_nextReact"
            states = ["S","I","R"]
            stateEvents = [["I"],["R"],[nothing]]
            eventHazards = [[beta], [alpha], [0.0]]
            hazardMultipliers = [["I"],[nothing],[nothing]]
            simpleHazards = true
            model! = gillespieNextReact_network!
            eventsPerState = Int64[1,1,1]


        elseif simType == "SIRD_nextReact"
            states = ["S","I","R","D"]
            stateEvents = [["I"],["R","D"],[nothing],[nothing]]
            eventHazards = [[beta], [alpha, gamma], [0.0],[0.0]]
            hazardMultipliers = [["I"],[nothing, nothing],[nothing],[nothing]]
            simpleHazards = true
            model! = gillespieNextReact_network!
            eventsPerState = Int64[1,2,1,1]
        end

        return states, stateEvents, eventHazards, hazardMultipliers, simpleHazards, model!, eventsPerState
    end


end
