#=
This is a code base for the main function of a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
=#
using Random, Conda, PyCall


# import required modules
push!( LOAD_PATH, "./" )    #Set path to current
using sirModels: gillespieDirect2Processes_rand, gillespieDirect2Processes_dist,
    gillespieFirstReact2Processes
using plotsPyPlot: plotSIRPyPlot

# main
function main(Display = true, save = true)

    # testing the gillespieDirect2Processes functions

    # Get same thing each time
    Random.seed!(1)

    # initialise variables
    N = [5, 10, 50, 100,1000,10000]

    # functions are not allowed to edit these arrays. Copy values in when passing
    # to the functions
    S_total = N .- 1
    I_total = ceil.(0.05 .* N)
    R_total = zeros(length(N))

    t_max = 200
    alpha = 0.4
    beta = 10 ./ N

    # Could reduce redundancy here too:

    # iterate through populations (Regular/rand)
    for i in 1:length(N)
        t, S, I, R = gillespieDirect2Processes_rand(t_max, copy(S_total[i]), copy(I_total[i]),
            copy(R_total[i]), alpha, beta[i], N[i])

        if Display | save
            population = N[i]
            outputFileName = "juliaGraphs/wellMixedDirectRandom/SIR_Model_Pop_$population"
            plotSIRPyPlot(t, [S, I, R], alpha, beta[i], N[i], outputFileName, Display, save)
        end
    end

    # iterate through populations (Dist)
    for i in 1:length(N)
        t, S, I, R = gillespieDirect2Processes_dist(t_max, copy(S_total[i]), copy(I_total[i]),
            copy(R_total[i]), alpha, beta[i], N[i])

        if Display | save
            population = N[i]
            outputFileName = "juliaGraphs/wellMixedDirectDist/SIR_Model_Pop_$population"
            plotSIRPyPlot(t, [S, I, R], alpha, beta[i], N[i], outputFileName, Display, save)
        end
    end

    # iterate through populations (First Reaction/Sep)
    for i in 1:length(N)
        t, S, I, R = gillespieFirstReact2Processes(t_max, copy(S_total[i]), copy(I_total[i]),
            copy(R_total[i]), alpha, beta[i], N[i])

        if Display | save
            population = N[i]
            outputFileName = "juliaGraphs/wellMixedFirstReact/SIR_Model_Pop_$population"
            plotSIRPyPlot(t, [S, I, R], alpha, beta[i], N[i], outputFileName, Display, save)
        end
    end

end

# main(Display, save)
main(true, false)
