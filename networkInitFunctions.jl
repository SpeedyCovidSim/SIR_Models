module networkInitFunctions

    using LightGraphs, MetaGraphs, StatsBase
    using sirModels: gillespieDirect_network!, gillespieFirstReact_network!
    using networkFunctions: incrementInfectedNeighbors!

    export initialiseNetwork!

    function initialiseNetwork!(network, infectionProp, simType, alpha, beta, gamma = 0)
        #=
        Inputs
        network       : a undirected graph from LightGraphs and MetaGraphs
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

        # initialise dictionaries and arrays for storing network attributes
        networkVertex_dict = Dict{Int64, Dict{String, Any}}()
        network_dict = Dict{String, Any}()
        stateTotals = convert.(Int, zeros(length(states)))


        # return num vertices
        numVertices = nv(network)

        # array containing whether or not the individual is "S"
        isS = convert.(Bool,zeros(numVertices))

        # Initialise all states as "S", all numInfectedNeighbors as zero
        # states[1] will always be S/the initial state
        # for loop has i as the vertex index
        for i in vertices(network)

            # use vertex index as the primary key for the dictionary
            networkVertex_dict[i] = Dict("state"=>states[1], "initInfectedNeighbors"=>0)
            isS[i] = true

        end

        # randomly choose individuals to be infected
        infectedVertices = sample(1:numVertices, Int(ceil(infectionProp *
            numVertices)), replace = false)

        # Set them as infected and increment numInfectedNeighbors for their localNeighbourhood
        # by 1
        for i in infectedVertices
            networkVertex_dict[i]["state"] = "I"
            isS[i] = false

            incrementInfectedNeighbors!(network, networkVertex_dict, i)

        end

        # count the num of vertices in each state
        for i in vertices(network)
            stateTotals += states .== networkVertex_dict[i]["state"]
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

        return networkVertex_dict, network_dict, stateTotals, isS, model!
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
        end

        return states, stateEvents, eventHazards, hazardMultipliers, simpleHazards, model!, eventsPerState
    end


end
