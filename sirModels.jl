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
        alpha   : probability of infected person recovering [0,1]
        beta    : probability of susceptible person being infected [0,1]

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

    function gillespieDirect2Processes_dist(t_max, S_total, I_total, R_total, alpha,
            beta, N, t_init = 0.0)
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
        alpha   : probability of infected person recovering [0,1]
        beta   : probability of susceptible person being infected [0,1]

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
        alpha   : probability of infected person recovering [0,1]
        beta   : probability of susceptible person being infected [0,1]

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

    function gillespieDirect2Processes_network(t_max, S_total, I_total, R_total, network, alpha,
            beta, N, t_init = 0.0)
        #=
        Note:
        Direct Gillespie Method, on network
        Directly samples from the exponential distribution

        Inputs
        t_init  : Initial time (default of 0)
        t_max   : Simulation end time
        S_total : Num people susceptible to infection
        I_total : Num people infected
        R_total : Num people recovered
        network : the network containing the population to operate on
        N       : Population size (unused)
        alpha   : probability of infected person recovering [0,1]
        beta    : probability of susceptible person being infected [0,1]

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
        states = ["S","I","R"]
        events = ["infected", "recovered"]

        while t[end] < t_max && I_total != 0
            # calculate the propensities to transition
            h_i = calcHazard(network, alpha, beta)
            h = sum(h_i)

            et = Exponential(1/h)

            # time to any event occurring
            delta_t = rand(et)
            #println(delta_t)

            # selection probabilities for each transition process. sum(j) = 1
            j = h_i ./ h

            # choose the index of the individual to transition
            eventIndex = sample(1:(S_total+I_total+R_total),pweights(j))

            # cause transition, change their state and incrementInfectedNeighbors
            # for their localNeighbourhood
            if get_prop(network, eventIndex, :state) == states[1]  # (S->I)
                event = events[1]
                changeState(network, eventIndex, states[2])
                incrementInfectedNeighbors(network, eventIndex, event)

                S_total -= 1
                I_total += 1

            elseif get_prop(network, eventIndex, :state) == states[2]
                    # (I->R) (assumes that I is not 0)
                event = events[2]
                changeState(network, eventIndex, states[3])
                incrementInfectedNeighbors(network, eventIndex, event)
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


end  # module sirModels
