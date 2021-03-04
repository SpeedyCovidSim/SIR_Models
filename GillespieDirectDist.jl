#=
This is a code base for a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
=#

#Pkg.add("Distributions")
#Pkg.add("Plots")
#Pkg.add("StatsBase)
using Distributions
using Plots
using Random
using StatsBase

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
function gillespieDirect2Processes(t_max, S_total, I_total, R_total, alpha,
        gamma, N, t_init = 0.0)

    # initialise outputs
    t = [copy(t_init)]
    S = [copy(S_total)]
    I = [copy(I_total)]
    R = [copy(R_total)]
    items = ["I","R"]

    while t[end] < t_max && I_total != 0
        # calculate the propensities to transition
        # h1 is propensity for infection, h2 is propensity for recovery
        h_i = [gamma * I_total * S_total, alpha * I_total]
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


#-----------------------------------------------------------------------------
#=
Inputs
t       : Array of times at which events have occured
SIR     : Array of arrays of Num people susceptible, infected and recovered at
            each t
N       : Population size

Outputs
png     : plot of SIR model over time [by default]
=#
function plots(t, SIR, N, Display=true, save=true)
    gr(reuse=true)

    plot(t, SIR, label=["Susceptible" "Infected" "Recovered"], lw = 2)
    #plot!(t, I, label="Infected", show = true)
    #display(plot!(t, R, label="Recovered", show = true))

    xlabel!("Time")
    ylabel!("Population Number")
    title!("SIR model over time with a population size of $N")
    plot!(size=(750,500))

    if Display
        # required to display graph on plots.
        display(plot!())
    end

    if save
        # Save graph as pngW
        png("juliaDistGraphs/SIR_Model_Pop_$N")
    end
end

#-----------------------------------------------------------------------------
# testing the gillespieDirect2Processes function

# Get same thing each time
Random.seed!(1)

# initialise variables
N = [10000]

S_total = N .- 1
I_total = N .* 0 .+ 1
R_total = N .* 0

t_max = 200
alpha = 0.4
gamma = 0.0004

# iterate through populations
for i in 1:length(N)
    t, S, I, R = gillespieDirect2Processes(t_max, S_total[i], I_total[i],
        R_total[i], alpha, gamma, N)

    plots(t, [S, I, R], N[i])

end
