
module networkFunctions
    using LightGraphs, MetaGraphs, StatsBase

    export calcHazard!, incrementInfectedNeighbors!, changeState!,
        incrementStateTotals!, outputStateTotals, calcHazardSir!, updateHazardSir!,
        calcHazardFirstReact!, updateHazardFirstReact!

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

    function calcHazardSir!(network, networkVertex_dict, network_dict) # also for SIRD

        # preallocate hazards array
        hazards = zeros(network_dict["population"])

        beta::Float64 = network_dict["S"]["eventHazards"][1]::Float64
        alpha::Float64 = sum(network_dict["I"]["eventHazards"])

        # calculate hazard at i
        # use multithreading to speed up
        #Threads.@threads for i in vertices(network)
        for i in vertices(network)
            if networkVertex_dict[i]["state"] === "S"
                hazards[i] = beta * networkVertex_dict[i]["initInfectedNeighbors"]
            elseif networkVertex_dict[i]["state"] === "I"
                hazards[i] = copy(alpha)

                #else # person is recovered or deceased and their hazard is zero
            end
        end
        return hazards
    end

    function updateHazardSir!(network, hazards, vertexIndex, prevState, newState,
        networkVertex_dict, network_dict, isS) # also works for SIRD

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
        #Threads.@threads for i in neighbors(network, vertexIndex)
        for i in neighbors(network, vertexIndex)
            # only update hazards for those that are susceptible
            # if networkVertex_dict[i]["state"] == "S"
            if isS[i]
                hazards[i] += increment
            end
        end

        return nothing
    end

    function calcHazardFirstReact!(network, networkVertex_dict, network_dict) # also for SIRD

        # # identify the number of possible events that could occur
        # # create a corresponding master event and hazard array for this - unpack
        # # the vector of vectors in the network_dict.
        # states::Array{String,1} = copy(network_dict["states"])
        # numPossEvents=0
        #
        # events = String[]
        # stateForEvents = String[]
        # eventHazards = Float64[]
        # allHazardMult = String[]
        #
        # for state in states
        #     # count how many possible events could occur for any individual
        #
        #     possEvents = network_dict[state]["events"]
        #     possMult = network_dict[state]["hazardMult"]
        #     numPossEvents += length(possEvents)
        #
        #     for i in 1:length(possEvents)
        #         if possEvents[i] === nothing
        #             push!(events, "")
        #         else
        #             push!(events, possEvents[i]::String)
        #         end
        #
        #         if possMult[i] === nothing
        #             push!(allHazardMult, "")
        #         else
        #             push!(allHazardMult, possMult[i])
        #         end
        #
        #         push!(eventHazards, network_dict[state]["eventHazards"][i])
        #         push!(stateForEvents, state)
        #     end
        # end
        #
        #
        # # preallocate hazards array
        # hazards = zeros(numPossEvents, network_dict["population"])
        #
        # # beta::Float64 = network_dict["S"]["eventHazards"][1]::Float64
        # # alpha::Float64 = network_dict["I"]["eventHazards"][1]::Float64
        #
        # # calculate hazard of events for individual i
        # # use multithreading to speed up
        # #Threads.@threads for i in vertices(network)
        # for i in vertices(network)
        #     for j in 1:numPossEvents
        #
        #         # determine whether the individual is an event's state
        #         if networkVertex_dict[i]["state"] === stateForEvents[j]
        #
        #             # check if there is a multiplier for that event
        #             # multiplier = allHazardMult[j]
        #
        #             if allHazardMult[j] != ""
        #                 if allHazardMult[j] == "I"
        #                     hazards[j,i] = eventHazards[j] *  networkVertex_dict[i]["initInfectedNeighbors"]
        #                 end
        #                 # add more here as becomes relevant
        #             else
        #                 hazards[j,i] = copy(eventHazards[j])
        #             end
        #         end
        #     end
        # end
        #
        # # hazards[:,i] = (networkVertex_dict[i]["state"] .=== stateForEvents) .* eventHazards
        #
        #
        # return hazards, events, stateForEvents, eventHazards, allHazardMult

        maxEvents::Int64 = maximum(network_dict["eventsPerState"]::Array{Int64,1})

        # preallocate hazards array
        hazards = zeros(network_dict["population"]::Int64 * maxEvents)

        # beta::Float64 = network_dict["S"]["eventHazards"][1]::Float64
        # alpha::Float64 = network_dict["I"]["eventHazards"][1]::Float64


        # calculate hazard of events for individual i
        # use multithreading to speed up
        #Threads.@threads for i in vertices(network)
        for i in vertices(network)

            # loop through all possible states for a given individual
            stateIndex = 1
            for numEvents in network_dict["eventsPerState"]::Array{Int64,1}

                state = network_dict["states"][stateIndex]::String

                # determine whether the individual is the relevant state
                if networkVertex_dict[i]["state"] === state

                    # if there a multiple events for a given state
                    for j in 1:numEvents

                        # check if there is a multiplier for that event
                        # multiplier = allHazardMult[j]

                        if !isnothing(network_dict[state]["hazardMult"][j])
                            if network_dict[state]["hazardMult"][j] === "I"

                                hazards[i + (j-1)*network_dict["population"]::Int64] =
                                    network_dict[state]["eventHazards"][j] *
                                    networkVertex_dict[i]["initInfectedNeighbors"]
                            end
                        else
                           hazards[i + (j-1)*network_dict["population"]::Int64] = copy(network_dict[state]["eventHazards"][j])
                        end

                        # if allHazardMult[j] != ""
                        #     if allHazardMult[j] == "I"
                        #
                        #         hazards[j,i] = eventHazards[j] *  networkVertex_dict[i]["initInfectedNeighbors"]
                        #     end
                        #     # add more here as becomes relevant
                        # else
                        #     hazards[j,i] = copy(eventHazards[j])
                        # end
                    end
                end
                stateIndex+=1
            end # events loop

        end # vertices loop

        # hazards[:,i] = (networkVertex_dict[i]["state"] .=== stateForEvents) .* eventHazards


        return hazards#, events, stateForEvents, eventHazards, allHazardMult

    end

    # function updateHazardFirstReact!(network, hazards, reaction_j, prevState, newState,
    #     networkVertex_dict, network_dict, isS, events, stateForEvents, eventHazards, allHazardMult)

    function updateHazardFirstReact!(network, hazards, reaction_j, prevState, newState,
        networkVertex_dict, network_dict, isS) # also works for SIRD

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
        for i in 1:eventsNewState
            hazards[reaction_j[2] + (i-1)*network_dict["population"]::Int64] = copy(network_dict[newState]["eventHazards"][i]::Float64)
        end


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
            for i in neighbors(network, reaction_j[2])

                # only update hazards for those that are susceptible
                #if networkVertex_dict[i]["state"] == "S"
                if isS[i]
                    hazards[i] += increment
                    # Make sure we don't have a precision error
                    if abs(hazards[i]) < 1e-10
                        hazards[i] = 0.0
                    end
                end

            end

        else

            # extract which states have hazards depending on neighbors
            stateIndex = 1
            stateIndex1 = Vector{Int64}[]
            stateIndex2 = Vector{Int64}[]
            for numEvents in network_dict["eventsPerState"]::Array{Int64,1}

                state = network_dict["states"][stateIndex]::String

                for j in 1:numEvents
                    # newState != prevState
                    if prevState === network_dict["states"]["hazardMult"][j]
                        push!(stateIndex1, copy([stateIndex, j]))
                    elseif newState === network_dict["states"]["hazardMult"][j]
                        push!(stateIndex2, copy([stateIndex, j]))
                    end
                end

                stateIndex += 1
            end

            # stateIndex1/2 store stateIndex in position 1 and reaction index
            # in position 2
            if stateIndex1 != Vector{Int64}[] || stateIndex2 != Vector{Int64}[]

                if stateIndex1 != Vector{Int64}[]

                    # multithread hazard calculation to speed up
                    # if num neighbors is low, this will actually slow it down
                    #Threads.@threads for i in neighbors(network, vertexIndex)
                    for i in neighbors(network, reaction_j[2])
                        # update hazards for relevant states
                        for j in stateIndex1

                            if networkVertex_dict[i]["state"] === network_dict["states"][j[1]]

                                hazards[i + (j[2]-1)*network_dict["population"]::Int64] -= network_dict[network_dict["states"][j[1]]]["eventHazards"][j[2]]

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
                    for i in neighbors(network, reaction_j[2])
                        # update hazards for relevant states
                        for j in stateIndex2

                            if networkVertex_dict[i]["state"] === network_dict["states"][j[1]]

                                hazards[i + (j[2]-1)*network_dict["population"]::Int64] += network_dict[network_dict["states"][j[1]]]["eventHazards"][j[2]]

                            end

                        end
                    end
                end
            end

        end


        # stateIndex1 = findall(prevState .== stateForEvents)
        # stateIndex2 = findall(newState .== stateForEvents)
        #
        # # Zero all hazards from previous state
        # for i in stateIndex1
        #     hazards[i, reaction_j[2]] = 0.0
        # end
        #
        # # UPDATE later if events other than S depend on multiple things
        # for i in stateIndex2
        #     hazards[i, reaction_j[2]] = eventHazards[i]
        # end

        # # by simple hazards I mean a set of hazards where S is the only thing
        # # that depends on other states, and only depends on I
        # simpleHazards = network_dict["simpleHazards"]::Bool
        # if simpleHazards
        #     # sneaky way of doing it
        #     if newState === "I"
        #         increment = 1.0
        #     else
        #         increment = -1.0
        #     end
        #
        #     # Threads.@threads for i in neighbors(network, reaction_j[2])
        #     for i in neighbors(network, reaction_j[2])
        #
        #         # only update hazards for those that are susceptible
        #         #if networkVertex_dict[i]["state"] == "S"
        #         if isS[i]
        #             hazards[1,i] += eventHazards[1] * increment
        #             # Make sure we don't have a precision error
        #             if abs(hazards[1,i]) < 1e-10
        #                 hazards[1,i] = 0.0
        #             end
        #         end
        #
        #     end
        #
        # else
        #
        #     # hazards of neighbors depend on the event if true
        #     if sum(allHazardMult .== "") != length(allHazardMult)
        #         stateIndex1 = 0
        #         stateIndex2 = 0
        #
        #         # findall returns [] if nothing
        #         stateIndex1 = findall(prevState .=== allHazardMult) #-1
        #
        #         stateIndex2 = findall(newState .=== allHazardMult) #+1
        #
        #         if stateIndex1 != []
        #
        #             # multithread hazard calculation to speed up
        #             # if num neighbors is low, this will actually slow it down
        #             #Threads.@threads for i in neighbors(network, vertexIndex)
        #             for i in neighbors(network, reaction_j[2])
        #                 # update hazards for relevant states
        #                 for j in stateIndex1
        #
        #                     if networkVertex_dict[i]["state"] === stateForEvents[j]
        #                         hazards[j,i] -= eventHazards[j]
        #
        #                         # Make sure we don't have a precision error
        #                         if abs(hazards[j,i]) < 1e-10
        #                             hazards[j,i] = 0.0
        #                         end
        #                     end
        #
        #                 end
        #             end
        #         end
        #
        #         if stateIndex2 != []
        #
        #             # multithread hazard calculation to speed up
        #             # if num neighbors is low, this will actually slow it down
        #             #Threads.@threads for i in neighbors(network, vertexIndex)
        #             for i in neighbors(network, reaction_j[2])
        #                 # update hazards for relevant states
        #                 for j in stateIndex2
        #
        #                     if networkVertex_dict[i]["state"] === stateForEvents[j]
        #                         hazards[j,i] += eventHazards[j]
        #                     end
        #
        #                 end
        #             end
        #         end
        #     end
        #
        # end
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
    function incrementInfectedNeighbors!(network, networkVertex_dict, vertexIndex)
        #=
        Inputs
        network            : a undirected graph from LightGraphs and MetaGraphs
                             containing our population of interest
        networkVertex_dict : a dictionary containing each vertex in network,
                             with their states and numInfectedNeighbors
        vertexIndex        : the individual (vertex) the event occured to

        Outputs
        nothing            : works in place on the networkVertex_dict
        =#

        #for i in neighbors(network, vertexIndex)
        Threads.@threads for i in neighbors(network, vertexIndex)
            networkVertex_dict[i]["initInfectedNeighbors"] += 1
        end
        return nothing
    end

    function changeState!(networkVertex_dict, vertexIndex, newState, isS)
        #=
        Inputs
        networkVertex_dict     : a dictionary containing our population of interest
        vertexIndex            : the individual (vertex) the state change occured to
        newState               : state to change to (string)

        Outputs
        nothing     : works in place on the networkVertex_dict
        =#

        networkVertex_dict[vertexIndex]["state"] = newState

        if newState != "S"
            isS[vertexIndex] = false
        end
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

        stateTotals[network_dict[prevState]["stateIndex"]]::Int64 -= 1
        stateTotals[network_dict[newState]["stateIndex"]]::Int64 += 1

        return nothing
    end

end
