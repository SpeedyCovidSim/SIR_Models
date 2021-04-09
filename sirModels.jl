#=
This is a code base for a simple SIR simulation using the Gillespie Direct
Method. Contains the simulation functions.

Author: Joel Trent and Josh Looker
=#

module sirModels

    using Distributions
    using Random
    using StatsBase
    using networkFunctions, LightGraphs, MetaGraphs

    function gillespieDirect2Processes_rand(t_max, S_total, I_total, R_total, alpha,
            beta, N, t_init = 0.0)
        #=
        Note:
        Direct Gillespie Method, Well Mixed
        Uses rand() to sample from the exponential distribution

        Inputs
        t_init  : Initial time (default of 0)
        t_max   : Simulation end time
        S_total : Num people susceptible to infection
        I_total : Num people infected
        R_total : Num people recovered
        N       : Population size (unused)
        alpha   : rate of infected person recovering
        beta    : rate of susceptible person being infected

        Outputs
        t       : Array of times at which events have occured
        S       : Array of Num people susceptible at each t
        I       : Array of Num people infected at each t
        R       : Array of Num people recovered at each t
        =#

        # initialise outputs
        t = [copy(t_init)]
        S = [copy(S_total)]
        I = [copy(I_total)]
        R = [copy(R_total)]

        while t[end] < t_max && I_total != 0
            # calculate the propensities to transition
            # h1 is propensity for infection, h2 is propensity for recovery
            h_i = [beta * I_total * S_total, alpha * I_total]
            h = sum(h_i)

            # time to any event occurring
            delta_t = -log(1-rand())/h
            #println(delta_t)

            # selection probabilities for each transition process. sum(j) = 1
            j = h_i ./ h

            # coding this way so can include more processes later with additional
            # elseif
            # could be done more efficiently if large number of processes
            choice = rand()

            if choice < j[1] && S_total != 0  # (S->I)
                S_total -= 1
                I_total += 1
            else    # (I->R) (assumes that I is not 0)
                I_total -= 1
                R_total += 1
            end

            push!(t, t[end] + delta_t)
            push!(S, copy(S_total))
            push!(I, copy(I_total))
            push!(R, copy(R_total))

        end # while

        return t, S, I , R
    end # function

    function gillespieDirect2Processes_dist(t_max::Float64, S_total::Int, I_total::Int, R_total::Int, alpha::Float64,
            beta::Float64, N::Int, t_init = 0.0)
        #=
        Note:
        Direct Gillespie Method, Well Mixed
        Directly samples from the exponential distribution

        Inputs
        t_init  : Initial time (default of 0)
        t_max   : Simulation end time
        S_total : Num people susceptible to infection
        I_total : Num people infected
        R_total : Num people recovered
        N       : Population size (unused)
        alpha   : rate of infected person recovering
        beta    : rate of susceptible person being infected

        Outputs
        t       : Array of times at which events have occured
        S       : Array of Num people susceptible at each t
        I       : Array of Num people infected at each t
        R       : Array of Num people recovered at each t
        =#

        # initialise outputs
        t = Float64[copy(t_init)]
        S = Int[copy(S_total)]
        I = Int[copy(I_total)]
        R = Int[copy(R_total)]
        items = ["I","R"]

        while t[end] < t_max && I_total != 0
            # calculate the propensities to transition
            # h1 is propensity for infection, h2 is propensity for recovery
            h_i = [beta * I_total * S_total, alpha * I_total]
            h = sum(h_i)

            et = Exponential(1/h)

            # time to any event occurring
            delta_t = rand(et)
            #println(delta_t)

            # selection probabilities for each transition process. sum(j) = 1
            j = h_i ./ h
            result = sample(items,pweights(j))

            # coding this way so can include more processes later with additional
            # elseif
            # could be done more efficiently if large number of processes
            if result == items[1] && S_total != 0  # (S->I)
                S_total -= 1
                I_total += 1
            else    # (I->R) (assumes that I is not 0)
                I_total -= 1
                R_total += 1
            end

            push!(t, t[end] + delta_t)
            push!(S, copy(S_total))
            push!(I, copy(I_total))
            push!(R, copy(R_total))

        end # while

        return t, S, I , R
    end # function

    function gillespieFirstReact2Processes(t_max, S_total, I_total, R_total, alpha,
            beta, N, t_init = 0.0)
        #=
        Note:
        First reaction Gillespie Method, Well Mixed

        Inputs
        t_init  : Initial time (default of 0)
        t_max   : Simulation end time
        S_total : Num people susceptible to infection
        I_total : Num people infected
        R_total : Num people recovered
        N       : Population size (unused)
        alpha   : rate of infected person recovering
        beta    : rate of susceptible person being infected

        Outputs
        t       : Array of times at which events have occured
        S       : Array of Num people susceptible at each t
        I       : Array of Num people infected at each t
        R       : Array of Num people recovered at each t
        =#
        # initialise outputs
        t = [copy(t_init)]
        S = [copy(S_total)]
        I = [copy(I_total)]
        R = [copy(R_total)]
        items = ["I","R"]

        while t[end] < t_max && I_total != 0
            # calculate the propensities to transition
            # h1 is propensity for infection, h2 is propensity for recovery
            h_i = [beta * I_total * S_total, alpha * I_total]
            h = sum(h_i)

            # delta t's for each process
            delta_tI = rand(Exponential(1/h_i[1]))
            delta_tR = rand(Exponential(1/h_i[2]))

            # coding this way so can include more processes later with additional
            # elseif
            # could be done more efficiently if large number of processes
            # event with lowest 'alarm clock' occurs first
            if delta_tI <= delta_tR && S_total != 0  # (S->I)
                S_total -= 1
                I_total += 1
                delta_t = delta_tI
            else    # (I->R) (assumes that I is not 0)
                I_total -= 1
                R_total += 1
                delta_t = delta_tR
            end

            push!(t, t[end] + delta_t)
            push!(S, copy(S_total))
            push!(I, copy(I_total))
            push!(R, copy(R_total))

        end # while

        return t, S, I , R
    end # function

    function gillespieDirect_network!(t_max, network, alpha, beta, N,
        networkVertex_dict, network_dict, stateTotals::Array{Int64,1}, isState, t_init = 0.0)
        #=
        Note:
        Direct Gillespie Method, on network
        Directly samples from the exponential distribution

        Inputs
        t_max   : Simulation end time
        network : the network containing the population to operate on
        alpha   : rate of infected person recovering
        beta    : rate of susceptible person being infected
        N       : Population size (unused)
        t_init  : Initial time (default of 0)

        Outputs
        t       : Array of times at which events have occured
        S       : Array of Num people susceptible at each t
        I       : Array of Num people infected at each t
        R       : Array of Num people recovered at each t
        =#

        # initialise outputs
        t = Float64[copy(t_init)]

        # could consider preallocating an array of arbitrary length to increase
        # speed and only switch to append once the end is reached
        # preallocate array about pop * numStates * num dependent processes/pathways (e.g. 2 for SIR and SIRD)
        numStates::Int64 = length(network_dict["states"])

        stateTotalsAll = convert.(Int64, zeros(network_dict["population"] * numStates * 2)::Array{Float64,1})

        stateTotalsAll[1:numStates] = copy(stateTotals)
        errorsCaught = 0 # this will allow us to know if we are not preallocating enough

        # calculate the propensities to transition
        # only calculate full array first time. Will cause speedups if network
        # is not fully connected
        h_i::Array{Float64,1} = calcHazardSir!(network, networkVertex_dict, network_dict, isState)

        iteration = 0

        #j = zeros(network_dict["population"])

        while t[end] < t_max && stateTotals[network_dict["I"]["stateIndex"]] != 0
            # including this in line in the loop rather than updating it in
            # incrementally like h_i. It made no difference to speed, so left it here
            # even for networks with a low degree of connection
            h = sum(h_i::Array{Float64,1})
            et = Exponential((1/h)::Float64)

            # time to any event occurring
            delta_t = rand(et)

            # selection probabilities for each transition process. sum(j) = 1

            # about 50% of the time was spent in this line - weights don't have
            # to be normalised
            #j = h_i ./ h

            # choose the index of the individual to transition
            vertexIndex::Int = sample((1:network_dict["population"])::UnitRange{Int64}, pweights(h_i::Array{Float64,1}))

            # cause transition, change their state

            # identify individual's state, w/ respect to property mapping in network description
            # let this be their previous state
            prevState::String = networkVertex_dict[vertexIndex]["state"]

            # determine num events that can happen to an individual in this state
            events::Array{String,1} = network_dict[prevState]["events"]

            eventIndex = 1
            # if multiple events possible, choose one based on hazard weighting
            if length(events) > 1
                # assumes constant probability of each event. Hazard does not
                # depend on anything but state
                # BAD ASSUMPTION, WHICH WE'LL DEAL WITH LATER
                eventHazardArray = network_dict[prevState]["eventHazards"]

                eventIndex = sample(1:length(events), pweights(eventHazardArray ./ sum(eventHazardArray)))
            end

            # change state of individual and note prev state and newState
            newState::String = events[eventIndex]

            # code like this in case of ghost processes
            if prevState != newState
                changeState!(networkVertex_dict, network_dict, vertexIndex, prevState, newState, isState)

                # update hazard of individual
                # update hazards of neighbors (if applicable)
                updateHazardSir!(network, h_i, vertexIndex, prevState, newState, networkVertex_dict, network_dict, isState)

                # increment stateTotals (network, prevState, newState)
                incrementStateTotals!(network, prevState, newState, stateTotals, network_dict)

                #incrementInfectedNeighbors!(network, networkVertex_dict, vertexIndex)
            end

            iteration +=1
            push!(t, t[end] + delta_t)

            # use preallocated array as much as possible
            try
                stateTotalsAll[(numStates*iteration+1):(numStates*iteration+numStates)] = copy(stateTotals)

            catch BoundsError # if run out of preallocation
                append!(stateTotalsAll, copy(stateTotals))
                errorsCaught = errorsCaught + 1
            end

        end # while

        if errorsCaught != 0
            println("Num Errors caught = $errorsCaught. Recommend increasing preallocation value")
        end

        return t, reshape(stateTotalsAll[1:numStates*iteration+numStates]::Array{Int64,1},length(network_dict["states"]),:)
    end # function

    function gillespieFirstReact_network!(t_max, network, alpha, beta, N,
        networkVertex_dict, network_dict, stateTotals::Array{Int64,1}, isState, t_init = 0.0)
        #=
        Note:
        First React Gillespie Method, on network
        Directly samples from the exponential distribution

        Inputs
        t_max   : Simulation end time
        network : the network containing the population to operate on
        alpha   : rate of infected person recovering
        beta    : rate of susceptible person being infected
        N       : Population size (unused)
        t_init  : Initial time (default of 0)

        Outputs
        t       : Array of times at which events have occured
        S       : Array of Num people susceptible at each t
        I       : Array of Num people infected at each t
        R       : Array of Num people recovered at each t
        =#

        # initialise outputs
        t = Float64[copy(t_init)]

        # could consider preallocating an array of arbitrary length to increase
        # speed and only switch to append once the end is reached
        # preallocate array about pop * numStates * num dependent processes/pathways (e.g. 2 for SIR and SIRD)
        numStates::Int64 = length(network_dict["states"])

        stateTotalsAll = convert.(Int64, zeros(network_dict["population"] * numStates * 2)::Array{Float64,1})

        stateTotalsAll[1:numStates] = copy(stateTotals)
        errorsCaught = 0 # this will allow us to know if we are not preallocating enough

        # calculate the propensities to transition
        # only calculate full array first time. Will cause speedups if network
        # is not fully connected

        # let h_i be a Float64 1D array of the hazards where it's size is the
        # population size multiplied by the max number of events that can happen
        # to any state. If only one event for a state, the hazard is stored in
        # the individual's network index in the array. If multiple, then they are
        # stored in individual's network index * 0, ... * 1,... etc. in the array
        h_i::Array{Float64,1} = calcHazardFirstReact!(network, networkVertex_dict, network_dict, isState)
        maxEvents::Int64 = maximum(network_dict["eventsPerState"]::Array{Int64,1})

        iteration = 0
        reaction_j = Int[0,0]
        while t[end] < t_max && stateTotals[network_dict["I"]["stateIndex"]] != 0

            if sum(h_i .< 0) > 0
                negElement = argmin(h_i)
                negState = networkVertex_dict[rem(negElement-1, network_dict["population"]::Int64)+1]["state"]
                println("An element of h_i is less than zero at iteration #$iteration !")
                println("It is in state $negState, with value $(h_i[negElement])")

            end

            # time to all events occurring
            delta_ti = rand.(Exponential.(1.0 ./h_i))

            # determine the first reaction that occurs
            reaction_index = argmin(delta_ti)

            # reaction_j[1] stores the index of the event that occurred for
            # a given state. If only one event per state, then it is one
            # reaction_j[2] stores the individual the event occurs to (vertexIndex)
            if maxEvents > 1
                reaction_j[1] = div(reaction_index-1, network_dict["population"]::Int64)+1
                reaction_j[2] = rem(reaction_index-1, network_dict["population"]::Int64)+1
            else
                reaction_j = [1, reaction_index]
            end

            # change state of individual and note prev state and newState
            prevState = networkVertex_dict[reaction_j[2]]["state"]::String
            newState = network_dict[prevState]["events"][reaction_j[1]]::String

            # cause transition, change their state

            # code like this in case of ghost processes
            if prevState != newState
                changeState!(networkVertex_dict, network_dict, reaction_j[2], prevState, newState, isState)

                # update hazard of individual
                # update hazards of neighbors (if applicable)
                updateHazardFirstReact!(network, h_i, reaction_j, prevState,
                    newState, networkVertex_dict, network_dict, isState)

                # increment stateTotals (network, prevState, newState)
                incrementStateTotals!(network, prevState, newState, stateTotals, network_dict)

                #incrementInfectedNeighbors!(network, networkVertex_dict, vertexIndex)
            end

            iteration +=1
            push!(t, t[end] + delta_ti[reaction_index])

            # use preallocated array as much as possible
            try
                stateTotalsAll[(numStates*iteration+1):(numStates*iteration+numStates)] = copy(stateTotals)

            catch BoundsError # if run out of preallocation
                append!(stateTotalsAll, copy(stateTotals))
                errorsCaught = errorsCaught + 1
            end

        end # while

        if errorsCaught != 0
            println("Num Errors caught = $errorsCaught. Recommend increasing preallocation value")
        end

        return t, reshape(stateTotalsAll[1:numStates*iteration+numStates]::Array{Int64,1},length(network_dict["states"]),:)
    end # function

    function gillespieNextReact_network!(t_max, network, alpha, beta, N,
        networkVertex_dict, network_dict, stateTotals::Array{Int64,1}, isState, t_init = 0.0)
        #=
        Note:
        Next React Gillespie Method, on network (See Gibson & Bruck 2000)
        Directly samples from the exponential distribution
        Uses absolute times

        Inputs
        t_max   : Simulation end time
        network : the network containing the population to operate on
        alpha   : rate of infected person recovering
        beta    : rate of susceptible person being infected
        N       : Population size (unused)
        t_init  : Initial time (default of 0)

        Outputs
        t       : Array of times at which events have occured
        S       : Array of Num people susceptible at each t
        I       : Array of Num people infected at each t
        R       : Array of Num people recovered at each t
        =#

        # initialise outputs
        t = Float64[copy(t_init)]

        # could consider preallocating an array of arbitrary length to increase
        # speed and only switch to append once the end is reached
        # preallocate array about pop * numStates * num dependent processes/pathways (e.g. 2 for SIR and SIRD)
        numStates::Int64 = length(network_dict["states"])

        stateTotalsAll = convert.(Int64, zeros(network_dict["population"] * numStates * 2)::Array{Float64,1})

        stateTotalsAll[1:numStates] = copy(stateTotals)
        errorsCaught = 0 # this will allow us to know if we are not preallocating enough

        # calculate the propensities to transition
        # only calculate full array first time. Will cause speedups if network
        # is not fully connected

        # let h_i be a Float64 1D array of the hazards where it's size is the
        # population size multiplied by the max number of events that can happen
        # to any state. If only one event for a state, the hazard is stored in
        # the individual's network index in the array. If multiple, then they are
        # stored in individual's network index * 0, ... * 1,... etc. in the array
        h_i::Array{Float64,1} = calcHazardFirstReact!(network, networkVertex_dict, network_dict)
        maxEvents::Int64 = maximum(network_dict["eventsPerState"]::Array{Int64,1})

        iteration = 0
        reaction_j = Int[0,0]
        while t[end] < t_max && stateTotals[network_dict["I"]["stateIndex"]] != 0

            if sum(h_i .< 0) > 0
                negElement = argmin(h_i)
                negState = networkVertex_dict[rem(negElement-1, network_dict["population"]::Int64)+1]["state"]
                println("An element of h_i is less than zero at iteration #$iteration !")
                println("It is in state $negState, with value $(h_i[negElement])")

            end

            # time to all events occurring
            delta_ti = rand.(Exponential.(1.0 ./h_i))

            # determine the first reaction that occurs
            reaction_index = argmin(delta_ti)

            # reaction_j[1] stores the index of the event that occurred for
            # a given state. If only one event per state, then it is one
            # reaction_j[2] stores the individual the event occurs to (vertexIndex)
            if maxEvents > 1
                reaction_j[1] = div(reaction_index-1, network_dict["population"]::Int64)+1
                reaction_j[2] = rem(reaction_index-1, network_dict["population"]::Int64)+1
            else
                reaction_j = [1, reaction_index]
            end

            # change state of individual and note prev state and newState
            prevState = networkVertex_dict[reaction_j[2]]["state"]::String
            newState = network_dict[prevState]["events"][reaction_j[1]]::String

            # cause transition, change their state

            # code like this in case of ghost processes
            if prevState != newState
                changeState!(networkVertex_dict, reaction_j[2], newState, isS)

                # update hazard of individual
                # update hazards of neighbors (if applicable)
                updateHazardFirstReact!(network, h_i, reaction_j, prevState,
                    newState, networkVertex_dict, network_dict, isS)

                # increment stateTotals (network, prevState, newState)
                incrementStateTotals!(network, prevState, newState, stateTotals, network_dict)

                #incrementInfectedNeighbors!(network, networkVertex_dict, vertexIndex)
            end

            iteration +=1
            push!(t, t[end] + delta_ti[reaction_index])

            # use preallocated array as much as possible
            try
                stateTotalsAll[(numStates*iteration+1):(numStates*iteration+numStates)] = copy(stateTotals)

            catch BoundsError # if run out of preallocation
                append!(stateTotalsAll, copy(stateTotals))
                errorsCaught = errorsCaught + 1
            end

        end # while

        if errorsCaught != 0
            println("Num Errors caught = $errorsCaught. Recommend increasing preallocation value")
        end

        return t, reshape(stateTotalsAll[1:numStates*iteration+numStates]::Array{Int64,1},length(network_dict["states"]),:)
    end # function

end  # module sirModels
