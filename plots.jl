#=
This is a code base for the plotting function of a simple SIR simulation
using the Gillespie Direct Method

Author: Joel Trent and Josh Looker
=#

#Pkg.add("Measures")

module plots

    using Plots, Measures

    function plotBenchmarks(tMean,tMedian, N, Display=true, save=true)
        #=
        Inputs
        tMean   : 2D array. Col 1 contains mean times for Julia function, Col 2 for
                  Python to complete simulation.
        tMedian : 2D array. Col 1 contains median times for Julia function, Col 2 for
                  Python to complete simulation.
        N       : Array of Population size used.

        Outputs
        png     : plot of SIR model over time [by default]
        =#

        gr(reuse=true)

        # MAY NEED TO USE EXP or log time if order of magnitude between times taken
        # need to use log N

        # Use margin to give white space around titling
        plot1 = plot(log10.(N), log10.(tMean), label=["Julia" "Python"], lw = 2, margin = 5mm)
        plot2 = plot(log10.(N), log10.(tMedian), label=["Julia" "Python"], lw = 2, margin = 5mm)
        plot(plot1, plot2, layout = (1, 2), title =
            ["Mean time to complete the Simulation" "Median time to complete the Simulation"],
            legend=:topleft)
        plot!(size=(1000,500))
        xlabel!("Log10 Population size used")
        ylabel!("Log10 Simulation time (log10(s))")

        if Display
            # required to display graph on plots.
            display(plot!())
        end

        if save
            # Save graph as pngW
            png("Benchmarks/SimulationTimes")
        end
    end

    function plotSIR(t, SIR, alpha, beta, N, outputFileName="plot", Display=true, save=true)
        #=
        Inputs
        t              : Array of times at which events have occured
        SIR            : Array of arrays of Num people susceptible, infected and recovered at
                         each t
        N              : Population size
        outputFileName : the name/location to save the plot as/in

        Outputs
        png     : plot of SIR model over time [by default]
        =#

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
            png("$outputFileName")
        end
    end

end  # module plots