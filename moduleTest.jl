#=
This is a test of the module functionality

This is a code base for a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
=#

module moduleTest

    export gillespieDirect2Processes


    #Pkg.add("Distributions")
    #Pkg.add("Plots")
    using Distributions
    using Plots
    using Random

    #-----------------------------------------------------------------------------
    #=
    Inputs
    t_init  : Initial time (default of 0)
    t_max   : Simulation end time
    S_total : Num people susceptible to infection
    I_total : Num people infected
    R_total : Num people recovered
    N       : Population size
    alpha   : probability of infected person recovering [0,1]
    gamma   : probability of susceptible person being infected [0,1]

    Outputs
    t       : Array of times at which events have occured
    S       : Array of Num people susceptible at each t
    I       : Array of Num people infected at each t
    R       : Array of Num people recovered at each t
    =#
    function gillespieDirect2Processes(t_max, S_total, I_total, R_total, alpha, gamma, N, t_init = 0.0)

        # initialise outputs
        t = [copy(t_init)]
        S = [copy(S_total)]
        I = [copy(I_total)]
        R = [copy(R_total)]

        while t[end] < t_max && I_total != 0
            # calculate the propensities to transition
            # h1 is propensity for infection, h2 is propensity for recovery
            h_i = [gamma * I_total * S_total, alpha * I_total]
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
end
