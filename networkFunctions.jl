
module networkFunctions
    using LightGraphs, MetaGraphs, StatsBase

    export initialiseNetwork!, calcHazard!, incrementInfectedNeighbors!, changeState!,
        incrementStateTotals!, outputStateTotals, calcHazardSir!, updateHazardSir!

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

        states, stateEvents, eventHazards, hazardMultipliers = simType!(simType, alpha, beta, gamma)

        # return num vertices
        numVertices = nv(network)

        # Initialise all states as "S", all numInfectedNeighbors as zero
        # states[1] will always be S/the initial state
        for i in vertices(network)
            set_props!(network, i, Dict(:initInfectedNeighbors=>0, :state=>states[1]))

        end

        # randomly choose individuals to be infected
        infectedVertices = sample(1:numVertices, Int(ceil(infectionProp *
            numVertices)), replace = false)

        # Set them as infected and increment numInfectedNeighbors for their localNeighbourhood
        # by 1
        for i in infectedVertices
            set_prop!(network, i, :state, "I")
            #set_prop!(network, i, :stateIndex, findfirst(states .== "I"))

            incrementInfectedNeighbors!(network, i)

            #=
            localNeighbourhood = neighbors(network, i)
            for j in localNeighbourhood
                set_prop!(network, j, :numInfectedNeighbors, get_prop(network, j, :numInfectedNeighbors) + 1)

                #= Technically only matters if they are susceptible, but will write
                like this^ to make more future proof (additional states)
                if  get_prop(network, j, :state) == "S"

                end=#
            end
            =#
        end

        #I_total = length(infectedVertices)
        #S_total = numVertices - I_total
        #R_total = 0

        stateTotals = convert.(Int, zeros(length(states)))
        #stateIncrementor = zeros(length(states))

        # count the num of vertices in each state
        for i in vertices(network)
            stateTotals += states .== get_prop(network, i, :state)
        end

        stateIndex = 1
        for state in states
            set_props!(network, Dict(Symbol(state, "Total")=>stateTotals[stateIndex],
                Symbol(state, "Events")=>stateEvents[stateIndex],
                Symbol(state, "EventHazards")=>eventHazards[stateIndex],
                Symbol(state, "HazardMult")=>hazardMultipliers[stateIndex]))
            stateIndex +=1
        end

        set_props!(network, Dict(:population=>numVertices, :states=>states))

        #=
        set_props!(network, Dict(:stateTotals=>stateTotals, :states=>states,
            :stateEvents=>stateEvents, :eventHazards=>eventHazards,
            :population=>numVertices, :stateIncrementor=>stateIncrementor,
            :hazardMultipliers=>hazardMultipliers))
        =#
        return
    end

    function simType!(simType, alpha, beta, gamma)
        #=
        Inputs
        simType       : type of sim to use. ["SIR", "SIRD"]
        alpha         : rate of infected person recovering
        beta          : rate of susceptible person being infected
        gamma         : rate of infected person dieing

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
            stateEvents = [["I"],["R"],nothing]
            eventHazards = [[beta], [alpha], [0]]
            hazardMultipliers = ["I",nothing,nothing]


        elseif simType == "SIRD_direct"
            states = ["S","I","R","D"]
            stateEvents = [["I"],["R","D"],nothing,nothing]
            eventHazards = [[beta], [alpha, gamma], [0],[0]]
            hazardMultipliers = ["I",nothing,nothing,nothing]

        end

        return states, stateEvents, eventHazards, hazardMultipliers
    end

    function calcHazard!(network, updateOnly = false, hazards = [],
        vertexIndex = 0, prevState = "S", newState = "I")
        #=
        Inputs
        network : a undirected graph from LightGraphs and MetaGraphs
                  containing our population of interest
        alpha   : rate of infected person recovering
        beta    : rate of susceptible person being infected
        hazards : calculated hazard for each individual (vertex) in the network
        updateOnly : bool, true means only update the hazards array
        vertexIndex : the individual (vertex) the event occured to


        event       : string containing either "recovered" or "infected"
        events      : an array of strings = ["infected", "recovered"]

        Outputs
        hazards : calculated hazard for each individual (vertex) in the network
                : or nothing if updateOnly as it operates in place on hazards array
        =#

        if updateOnly
            # change hazard of affected individual
            # if infected, hazard is alpha, otherwise 0 as they've recovered
            # NOT WRITTEN TO ALLOW FOR MULTIPLIER
            multiplier = get_prop(network, :hazardMultipliers)[get_prop(network,vertexIndex, :stateIndex)]
            if isnothing(multiplier)
                hazards[vertexIndex] = sum(get_prop(network, :eventHazards)[get_prop(network,vertexIndex, :stateIndex)])

            elseif  multiplier == "I" # multiplier in use
                hazards[vertexIndex] = sum(get_prop(network, :eventHazards)[get_prop(network,vertexIndex, :stateIndex)]
                                        * get_prop(network, vertexIndex, :initInfectedNeighbors))

                # will need to add additional functionality to this later
            end

            allMultipliers = get_prop(network, :hazardMultipliers)

            # hazards of neighbors depend on the event if true
            if sum(isnothing(allMultipliers)) != length(allMultipliers)
                stateIndex1 = 0
                stateIndex2 = 0

                # findall returns [] if nothing
                stateIndex1 = findall(prevState .== allMultipliers) #-1

                stateIndex2 = findall(newState .== allMultipliers) #+1

                # determine which multipliers aren't nothing
                #multIndexes = findall(!isnothing(allMultipliers))

                if stateIndex1 != []
                    # multithread hazard calculation to speed up
                    # if num neighbors is low, this will actually slow it down
                    #Threads.@threads for i in neighbors(network, vertexIndex)
                    for i in neighbors(network, vertexIndex)
                        # only update hazards for those that are susceptible


                        # THIS WILL BREAK IF HAZARD IS DEPENDENT ON MULTIPLE MULTIPLIERS. OR IF MULTIPLIER SHOULD BE INVERSE SIGN
                        # OR IF MULTIPLE HAZARDS ASSOCIATED WITH A STATE
                        if get_prop(network, i, :stateIndex) in stateIndex1
                            hazards[i] = hazards[i] - get_prop(network, :eventHazards)[get_prop(network, i, :stateIndex)][1]
                        end
                    end
                end

                if stateIndex2 != []
                    # multithread hazard calculation to speed up
                    # if num neighbors is low, this will actually slow it down
                    #Threads.@threads for i in neighbors(network, vertexIndex)
                    for i in neighbors(network, vertexIndex)
                        # only update hazards for those that are susceptible


                        # THIS WILL BREAK IF HAZARD IS DEPENDENT ON MULTIPLE MULTIPLIERS. OR IF MULTIPLIER SHOULD BE INVERSE SIGN
                        # OR IF MULTIPLE HAZARDS ASSOCIATED WITH A STATE
                        if get_prop(network, i, :stateIndex) in stateIndex2
                            hazards[i] = hazards[i] + get_prop(network, :eventHazards)[get_prop(network, i, :stateIndex)][1]
                        end
                    end
                end

            end


            #fil1 = filter_vertices(network[neighbors(network, vertexIndex)], :state, "S")

        else # initialisation

            # preallocate hazards array
            hazards = zeros(get_prop(network, :population))

            # calculate hazard at i
            # use multithreading to speed up
            #Threads.@threads for i in vertices(network)
            for i in vertices(network)
                # if multiple hazards for the state, sum them
                eventHazards = get_prop(network, :eventHazards)[get_prop(network,i, :stateIndex)]

                # check if there is a multiplier
                multiplier = get_prop(network, :hazardMultipliers)[get_prop(network,i, :stateIndex)]
                if !isnothing(multiplier)
                    if multiplier == "I"
                        eventHazards = eventHazards * get_prop(network, i, :initInfectedNeighbors)
                    end
                    # add more here as becomes relevant
                end

                hazards[i] = sum(eventHazards)


                #else # person is recovered and their hazard is zero
            end
            return hazards
        end

        return nothing
    end

    function calcHazardSir!(network) # also for SIRD

        # preallocate hazards array
        hazards = zeros(get_prop(network, :population))

        beta = get_prop(network, :SEventHazards)[1]
        alpha = get_prop(network, :IEventHazards)[1]

        # calculate hazard at i
        # use multithreading to speed up
        Threads.@threads for i in vertices(network)
            if get_prop(network, i, :state) == "S"
                hazards[i] = beta * get_prop(network, i, :initInfectedNeighbors)
            elseif get_prop(network, i, :state) == "I"
                hazards[i] = copy(alpha)

                #else # person is recovered or deceased and their hazard is zero
            end
        end
        return hazards
    end

    function updateHazardSir!(network, hazards, vertexIndex, prevState, newState) # also works for SIRD

        beta = get_prop(network, :SEventHazards)[1]
        #alpha = get_prop(network, :IHazard)

        # change hazard of affected individual
        # if infected, hazard is alpha, otherwise 0 as they've recovered
        hazards[vertexIndex] = get_prop(network, Symbol(newState, "EventHazards"))[1]

        #alpha * (event == "I")

        if newState == "I"
            increment = 1 * beta
        else
            increment = -1 * beta
        end

        # multithread hazard calculation to speed up
        # if num neighbors is low, this will actually slow it down
        Threads.@threads for i in neighbors(network, vertexIndex)
            # only update hazards for those that are susceptible
            if get_prop(network, i, :state) == "S"
                hazards[i] = hazards[i] + increment
            end
        end

        return nothing
    end

    function calcHazardOld!(network, alpha, beta, updateOnly = false, hazards = [],
        vertexIndex = 0, event = "infected", events = ["I", "R"])
        #=
        Inputs
        network : a undirected graph from LightGraphs and MetaGraphs
                  containing our population of interest
        alpha   : rate of infected person recovering
        beta    : rate of susceptible person being infected
        hazards : calculated hazard for each individual (vertex) in the network
        updateOnly : bool, true means only update the hazards array
        vertexIndex : the individual (vertex) the event occured to
        event       : string containing either "recovered" or "infected"
        events      : an array of strings = ["infected", "recovered"]

        Outputs
        hazards : calculated hazard for each individual (vertex) in the network
                : or nothing if updateOnly as it operates in place on hazards array
        =#

        if updateOnly
            # change hazard of affected individual
            # if infected, hazard is alpha, otherwise 0 as they've recovered
            hazards[vertexIndex] = alpha * (event == events[1])

            if event == events[2]
                increment = -1 * beta
            else
                increment = 1 * beta
            end

            # multithread hazard calculation to speed up
            # if num neighbors is low, this will actually slow it down
            Threads.@threads for i in neighbors(network, vertexIndex)
                # only update hazards for those that are susceptible
                if get_prop(network, i, :state) == "S"
                    hazards[i] = hazards[i] + increment
                end
            end

            #fil1 = filter_vertices(network[neighbors(network, vertexIndex)], :state, "S")

        else # initialisation

            # preallocate hazards array
            hazards = zeros(get_prop(network, :population))

            # calculate hazard at i
            # use multithreading to speed up
            Threads.@threads for i in vertices(network)
                if get_prop(network, i, :state) == "S"
                    hazards[i] = beta * get_prop(network, i, :initInfectedNeighbors)
                elseif get_prop(network, i, :state) == "I"
                    hazards[i] = copy(alpha)

                    #else # person is recovered and their hazard is zero
                end
            end
            return hazards
        end

        return nothing
    end

    function incrementInfectedNeighbors!(network, vertexIndex)
        #=
        Inputs
        network     : a undirected graph from LightGraphs and MetaGraphs
                      containing our population of interest
        vertexIndex : the individual (vertex) the event occured to
        event       : string containing either "recovered" or "infected"
        events      : an array of strings = ["infected", "recovered"]

        Outputs
        nothing     : works in place on the network
        =#

        for i in neighbors(network, vertexIndex)
            set_prop!(network, i, :initInfectedNeighbors, get_prop(network, i, :initInfectedNeighbors) + 1)
        end
        return nothing
    end

    function changeState!(network, vertexIndex, state)
        #=
        Inputs
        network     : a undirected graph from LightGraphs and MetaGraphs
                      containing our population of interest
        vertexIndex : the individual (vertex) the state change occured to
        state       : state to change to

        Outputs
        nothing     : works in place on the network
        =#

        set_prop!(network, vertexIndex, :state, state)
        #set_prop!(network, vertexIndex, :stateIndex, newStateIndex)

        #=
        if state == "I"
            set_prop!(network, vertexIndex, :state, "I")
        elseif state == "R"
            set_prop!(network, vertexIndex, :state, "R")
        end
        =#
        return nothing
    end

    function incrementStateTotals!(network, prevState, newState)
        #=
        Inputs
        network         : a undirected graph from LightGraphs and MetaGraphs
                          containing our population of interest
        prevStateIndex  : the previous state of the individual updated
        newStateIndex   : the new state of the individual updated

        Outputs
        nothing     : works in place on the network
        =#

        # stateIncrementor is an array of zeros
        #incrementorArray = copy(get_prop(network, :stateIncrementor))
        #incrementorArray[prevStateIndex] = -1
        #incrementorArray[newStateIndex] = 1

        prevStateSymbol = Symbol(prevState, "Total")
        newStateSymbol = Symbol(newState, "Total")

        set_props!(network, Dict(prevStateSymbol=>get_prop(network, prevStateSymbol) - 1,
            newStateSymbol=>get_prop(network, newStateSymbol) + 1))

        return nothing
    end

    function outputStateTotals(network, numStates)

        stateTotal = zeros(numStates)
        i = 1
        for state in get_prop(network, :states)
            stateTotal[i] = get_prop(network, Symbol(state, "Total"))
            i+=1
        end

        return stateTotal
    end
end
