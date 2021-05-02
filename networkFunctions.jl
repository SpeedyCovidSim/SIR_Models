
module networkFunctions
    using LightGraphs, StatsBase, Distributions, Random, TrackingHeaps#, MetaGraphs

    export calcHazard!, incrementInfectedNeighbors!, changeState!,
        incrementStateTotals!, outputStateTotals, calcHazardSir!, updateHazardSir!,
        calcHazardFirstReact!, updateHazardFirstReact!, updateNextReact!

    function calcHazard!(network, updateOnly = false, hazards = Float64[],
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

    function calcHazardSir!(network, networkVertex_df, network_dict, isState) # also for SIRD

        # preallocate hazards array
        hazards = zeros(network_dict["population"])

        beta::Float64 = network_dict["S"]["eventHazards"][1]::Float64
        alpha::Float64 = sum(network_dict["I"]["eventHazards"])

        # calculate hazard at i
        # use multithreading to speed up
        #Threads.@threads for i in vertices(network)
        for i::Int64 in vertices(network)
            if isState[i, network_dict["S"]["stateIndex"]::Int64] #networkVertex_df[i, :state] === "S"
                hazards[i] = (beta * networkVertex_df[i, :initInfectedNeighbors]) / networkVertex_df[i, :connectivity]
            elseif isState[i, network_dict["I"]["stateIndex"]::Int64] #networkVertex_df[i, :state] === "I"
                hazards[i] = copy(alpha)

                #else # person is recovered or deceased and their hazard is zero
            end
        end
        return hazards
    end

    function updateHazardSir!(network, hazards, vertexIndex, prevState, newState,
        networkVertex_df, network_dict, isState) # also works for SIRD

        beta::Float64 = network_dict["S"]["eventHazards"][1]::Float64

        # change hazard of affected individual
        hazards[vertexIndex] = sum(network_dict[newState]["eventHazards"])

        increment::Float64 = 0.0

        if newState === "I"
            increment = 1.0 * beta
        else
            increment = -1.0 * beta
        end

        # multithread hazard calculation to speed up
        # if num neighbors is low, this will actually slow it down
        # Threads.@threads for i in neighbors(network, vertexIndex)
        for i::Int64 in neighbors(network, vertexIndex)
            # only update hazards for those that are susceptible
            # if networkVertex_df[i, :state] == "S"
            if isState[i, 1] #network_dict["S"]["stateIndex"]::Int64]
                hazards[i] += increment / networkVertex_df[i, :connectivity]
            end
        end

        return nothing
    end

    function calcHazardFirstReact!(network, networkVertex_df, network_dict, isState) # also for SIRD

        # identify the max number of possible events
        maxEvents::Int64 = maximum(network_dict["eventsPerState"]::Array{Int64,1})

        # preallocate hazards array
        hazards = zeros(network_dict["population"]::Int64 * maxEvents)

        # calculate hazard of events for individual i
        # use multithreading to speed up
        #Threads.@threads for i in vertices(network)
        for i::Int64 in vertices(network)

            # loop through all possible states for a given individual
            stateIndex = 1
            for numEvents in network_dict["eventsPerState"]::Array{Int64,1}

                state = network_dict["states"][stateIndex]::String

                # determine whether the individual is the relevant state
                if isState[i, network_dict[state]["stateIndex"]::Int64] #networkVertex_df[i, :state] === state

                    # if there a multiple events for a given state
                    for j::Int64 in 1:numEvents

                        # check if there is a multiplier for that event
                        # multiplier = allHazardMult[j]

                        if !isnothing(network_dict[state]["hazardMult"][j])
                            if network_dict[state]["hazardMult"][j] === "I"

                                hazards[i + (j-1)*network_dict["population"]::Int64] =
                                    (network_dict[state]["eventHazards"][j] *
                                    networkVertex_df[i, :initInfectedNeighbors]) / networkVertex_df[i, :connectivity]
                            end
                        else
                           hazards[i + (j-1)*network_dict["population"]::Int64] = copy(network_dict[state]["eventHazards"][j])
                        end

                    end
                end
                stateIndex+=1
            end # events loop

        end # vertices loop

        return hazards

    end

    function updateIndivHazardFirstReact!(hazards, reaction_j, newState,
        prevState, network_dict)

        # reaction_j position 1 contains reaction index, position 2 contains individual
        # change hazard of affected individual. Zero all hazards from previous state
        stateIndex1 = 0
        stateIndex2 = 0

        stateIndex1 = network_dict[prevState]["stateIndex"]::Int64
        stateIndex2 = network_dict[newState]["stateIndex"]::Int64

        # determine num events for these states
        eventsPrevState = network_dict["eventsPerState"][stateIndex1]::Int64
        eventsNewState = network_dict["eventsPerState"][stateIndex2]::Int64

        # Zero all hazards from previous state that won't be overwritten by new values
        if eventsPrevState > eventsNewState

            # logic for only allowing hazards to be zeroed from events greater
            # than the num events in new state
            for i in (1 + (eventsPrevState - (eventsPrevState-eventsNewState))):eventsPrevState

                hazards[reaction_j[2] + (i-1)*network_dict["population"]::Int64] = 0.0
            end
        end

        # UPDATE later if events other than S depend on multiple things
        for i::Int64 in 1:eventsNewState
            hazards[reaction_j[2] + (i-1)*network_dict["population"]::Int64] =
                copy(network_dict[newState]["eventHazards"][i]::Float64)
        end
        return nothing
    end

    function getDependentStates!(network_dict, prevState, newState)
        stateIndex = 1
        stateIndex1 = Vector{Int64}[]
        stateIndex2 = Vector{Int64}[]

        # numEvents is the number of events for each state, iteratively
        for numEvents::Int64 in network_dict["eventsPerState"]::Array{Int64,1}

            state = network_dict["states"][stateIndex]::String

            # j is 1:numEvents in a given state
            for j::Int64 in 1:numEvents
                # newState != prevState
                if prevState === network_dict[state]["hazardMult"][j]
                    push!(stateIndex1, copy([stateIndex, j]))
                elseif newState === network_dict[state]["hazardMult"][j]
                    push!(stateIndex2, copy([stateIndex, j]))
                end
            end

            stateIndex += 1
        end
        return stateIndex1, stateIndex2
    end

    function updateNeighborHazardFirstReact!(network, hazards, reaction_j, prevState, newState,
        networkVertex_df, network_dict, isState)

        # by simple hazards I mean a set of hazards where S is the only thing
        # that depends on other states, and only depends on I
        simpleHazards = network_dict["simpleHazards"]::Bool
        if simpleHazards

            beta::Float64 = network_dict["S"]["eventHazards"][1]::Float64

            # sneaky way of doing it
            if newState === "I"
                increment = 1.0 * beta
            else
                increment = -1.0 * beta
            end

            # Threads.@threads for i in neighbors(network, reaction_j[2])
            for i::Int64 in neighbors(network, reaction_j[2])

                # only update hazards for those that are susceptible
                #if networkVertex_df[i, :state] == "S"
                if isState[i, 1] #network_dict["S"]["stateIndex"]::Int64]
                    hazards[i] += increment / networkVertex_df[i, :connectivity]
                    # Make sure we don't have a precision error
                    if abs(hazards[i]) < 1e-10
                        hazards[i] = 0.0
                    end
                end

            end
            return nothing
        end

        # -----------------------------------------------------------------

        # extract which states have hazards depending on neighbors
        stateIndex1, stateIndex2 = getDependentStates!(network_dict, prevState, newState)

        # stateIndex1/2 store stateIndex in position 1 and reaction index
        # in position 2

        if stateIndex1 != Vector{Int64}[]

            # multithread hazard calculation to speed up
            # if num neighbors is low, this will actually slow it down
            #Threads.@threads for i in neighbors(network, vertexIndex)
            for i::Int64 in neighbors(network, reaction_j[2])
                # update hazards for relevant states
                for j::Array{Int64,1} in stateIndex1

                    #if networkVertex_df[i, :state]::String === network_dict["states"][j[1]]::String
                    if isState[i, j[1]]
                        @inbounds hazards[i +
                            (j[2]-1)*network_dict["population"]::Int64] -=
                            network_dict[networkVertex_df[i, :state]::String]["eventHazards"][j[2]]::Float64 /
                            networkVertex_df[i, :connectivity]

                        # Make sure we don't have a precision error
                        if abs(hazards[i + (j[2]-1)*network_dict["population"]::Int64]) < 1e-10
                            hazards[i + (j[2]-1)*network_dict["population"]::Int64] = 0.0
                        end
                    end
                end
            end
        end

        if stateIndex2 != Vector{Int64}[]

            # multithread hazard calculation to speed up
            # if num neighbors is low, this will actually slow it down
            #Threads.@threads for i in neighbors(network, vertexIndex)
            for i::Int64 in neighbors(network, reaction_j[2])
                # update hazards for relevant states
                for j::Array{Int64,1} in stateIndex2

                    #if networkVertex_df[i, :state]::String === network_dict["states"][j[1]]::String
                    if isState[i, j[1]]
                        @inbounds hazards[i +
                        (j[2]-1)*network_dict["population"]::Int64] +=
                        network_dict[networkVertex_df[i, :state]::String]["eventHazards"][j[2]]::Float64 /
                        networkVertex_df[i, :connectivity]

                    end
                end
            end
        end

        # ----------------------------------------------------------------------
        return nothing
    end

    function updateHazardFirstReact!(network, hazards, reaction_j, prevState, newState,
        networkVertex_df, network_dict, isState)

        # update hazard for individual the event happened to -------------------
        updateIndivHazardFirstReact!(hazards, reaction_j, newState,
            prevState, network_dict)

        # update hazard for neighbors of the individual the event happened to --
        updateNeighborHazardFirstReact!(network, hazards, reaction_j, prevState, newState,
            networkVertex_df, network_dict, isState)

        return nothing
    end

    function updateIndivNextReact!(hazards, times_heap, currentTime, reaction_j, newState,
        prevState, network_dict)

        # reaction_j position 1 contains reaction index, position 2 contains individual
        # change hazard of affected individual. Zero all hazards from previous state
        stateIndex1 = 0
        stateIndex2 = 0
        index::Int64 = 0

        stateIndex1 = network_dict[prevState]["stateIndex"]::Int64
        stateIndex2 = network_dict[newState]["stateIndex"]::Int64

        # determine num events for these states
        eventsPrevState = network_dict["eventsPerState"][stateIndex1]::Int64
        eventsNewState = network_dict["eventsPerState"][stateIndex2]::Int64

        # Zero all hazards from previous state that won't be overwritten by new values
        # Set time to reaction to Inf
        if eventsPrevState > eventsNewState

            # logic for only allowing hazards to be zeroed from events greater
            # than the num events in new state
            for i::Int64 in (1 + (eventsPrevState - (eventsPrevState-eventsNewState))):eventsPrevState
                index = reaction_j[2] + (i-1)*network_dict["population"]::Int64

                hazards[index] = 0.0
                update!(times_heap, index, Inf)
            end
        end

        # UPDATE later if events other than S depend on multiple things
        for i::Int64 in 1:eventsNewState
            index = reaction_j[2] + (i-1)*network_dict["population"]::Int64
            hazards[index] = copy(network_dict[newState]["eventHazards"][i]::Float64)

            update!(times_heap, index, currentTime + rand(Exponential(1.0 /hazards[index])) )
        end
        return nothing
    end

    function updateNeighborNextReact!(network, hazards, times_heap, currentTime,
        reaction_j, prevState, newState, networkVertex_df, network_dict, isState)

        newHazard::Float64 = 0.0

        # by simple hazards I mean a set of hazards where S is the only thing
        # that depends on other states, and only depends on I
        simpleHazards = network_dict["simpleHazards"]::Bool
        if simpleHazards

            beta::Float64 = network_dict["S"]["eventHazards"][1]::Float64

            # sneaky way of doing it
            if newState === "I"
                increment = 1.0 * beta
            else
                increment = -1.0 * beta
            end

            newHazard = 0.0

            # Threads.@threads for i in neighbors(network, reaction_j[2])
            for i::Int64 in neighbors(network, reaction_j[2])

                # only update hazards for those that are susceptible
                #if networkVertex_df[i, :state] == "S"
                if isState[i, 1] #network_dict["S"]["stateIndex"]::Int64]

                    newHazard = hazards[i] + increment / networkVertex_df[i, :connectivity]

                    # Make sure we don't have a precision error
                    if abs(newHazard) < 1e-10
                        newHazard = 0.0
                    end


                    if getindex(times_heap, i) === Inf # have to redraw time to reaction
                        update!(times_heap, i, currentTime + rand(Exponential(1.0 /newHazard)) )
                        # println("I'm never called")
                    else
                        update!(times_heap, i, (hazards[i]/newHazard) * (getindex(times_heap,i)-currentTime) + currentTime)
                    end

                    hazards[i] = copy(newHazard)
                end

            end
            return nothing
        end

        # -----------------------------------------------------------------

        # extract which states have hazards depending on neighbors
        stateIndex1, stateIndex2 = getDependentStates!(network_dict, prevState, newState)

        # stateIndex1/2 store stateIndex in position 1 and reaction index
        # in position
        index::Int64 = 0

        if stateIndex1 != Vector{Int64}[]

            # multithread hazard calculation to speed up
            # if num neighbors is low, this will actually slow it down
            #Threads.@threads for i in neighbors(network, vertexIndex)
            for i::Int64 in neighbors(network, reaction_j[2])
                # update hazards for relevant states
                for j::Array{Int64,1} in stateIndex1

                    #if networkVertex_df[i, :state]::String === network_dict["states"][j[1]]::String
                    if isState[i, j[1]]
                        index = i + (j[2]-1)*network_dict["population"]::Int64

                        @inbounds newHazard = hazards[index] -
                            network_dict[networkVertex_df[i, :state]::String]["eventHazards"][j[2]]::Float64 /
                            networkVertex_df[i, :connectivity]

                        # Make sure we don't have a precision error
                        if abs(newHazard) < 1e-10
                            newHazard = 0.0
                        end

                        if getindex(times_heap, index) === Inf # have to redraw time to reaction
                            update!(times_heap, index, currentTime + rand(Exponential(1.0 /newHazard)) )
                            # println("I'm never called")
                        else
                            update!(times_heap, index, (hazards[index]/newHazard) * (getindex(times_heap,index)-currentTime) + currentTime)
                        end
                        hazards[index] = copy(newHazard)
                    end
                end
            end
        end

        if stateIndex2 != Vector{Int64}[]

            # multithread hazard calculation to speed up
            # if num neighbors is low, this will actually slow it down
            #Threads.@threads for i in neighbors(network, vertexIndex)
            for i::Int64 in neighbors(network, reaction_j[2])
                # update hazards for relevant states
                for j::Array{Int64,1} in stateIndex2

                    #if networkVertex_df[i, :state]::String === network_dict["states"][j[1]]::String
                    if isState[i, j[1]]

                        index = i + (j[2]-1)*network_dict["population"]::Int64

                        @inbounds newHazard = hazards[index] +
                            network_dict[networkVertex_df[i, :state]::String]["eventHazards"][j[2]::Int]::Float64 /
                            networkVertex_df[i, :connectivity]

                        if getindex(times_heap, index) === Inf # have to redraw time to reaction
                            update!(times_heap, index, currentTime + rand(Exponential(1.0 /newHazard)) )
                            # println("I'm never called")
                        else
                            update!(times_heap, index, (hazards[index]/newHazard) * (getindex(times_heap,index)-currentTime) + currentTime)
                        end

                        hazards[index] = copy(newHazard)
                    end
                end
            end
        end

        # ----------------------------------------------------------------------
        return nothing
    end


    function updateNextReact!(network, hazards, times_heap, currentTime, reaction_j, prevState, newState,
        networkVertex_df, network_dict, isState) # also works for SIRD

        # update hazard for individual the event happened to -------------------
        updateIndivNextReact!(hazards, times_heap, currentTime, reaction_j, newState,
            prevState, network_dict)

        # update hazard for neighbors of the individual the event happened to --
        updateNeighborNextReact!(network, hazards, times_heap, currentTime,
            reaction_j, prevState, newState, networkVertex_df, network_dict, isState)

        return nothing
    end


    #=
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
    =#
    function incrementInfectedNeighbors!(network, networkVertex_df, vertexIndex)
        #=
        Inputs
        network            : a undirected graph from LightGraphs and MetaGraphs
                             containing our population of interest
        networkVertex_df : a dataframe containing each vertex in network,
                             with their states and numInfectedNeighbors
        vertexIndex        : the individual (vertex) the event occured to

        Outputs
        nothing            : works in place on the networkVertex_df
        =#

        #for i in neighbors(network, vertexIndex)
        Threads.@threads for i::Int64 in neighbors(network, vertexIndex)
            networkVertex_df[i, :initInfectedNeighbors] += 1
        end
        return nothing
    end

    function changeState!(networkVertex_df, network_dict, vertexIndex::Int64, prevState, newState, isState::BitArray{2})
        #=
        Inputs
        networkVertex_df     : a dataframe containing our population of interest
        vertexIndex            : the individual (vertex) the state change occured to
        newState               : state to change to (string)

        Outputs
        nothing     : works in place on the networkVertex_df
        =#

        networkVertex_df[vertexIndex, :state] = newState

        isState[vertexIndex, network_dict[prevState]["stateIndex"]::Int64] = false
        isState[vertexIndex, network_dict[newState]["stateIndex"]::Int64] = true

        # if newState != "S"
        #     isS[vertexIndex] = false
        # end
        return nothing
    end

    function incrementStateTotals!(network, prevState, newState, stateTotals, network_dict)
        #=
        Inputs
        network         : a undirected graph from LightGraphs and MetaGraphs
                          containing our population of interest
        prevState  : the previous state of the individual updated (string)
        newState   : the new state of the individual updated (string)

        Outputs
        nothing     : works in place on the network
        =#

        stateTotals[network_dict[prevState]["stateIndex"]::Int64]::Int64 -= 1
        stateTotals[network_dict[newState]["stateIndex"]::Int64]::Int64 += 1

        return nothing
    end

end
