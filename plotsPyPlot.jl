#=
This is a code base for the plotting function of a simple SIR simulation
using the Gillespie Direct Method. Using PyPlot

Author: Joel Trent and Josh Looker
=#


module plotsPyPlot

    using PyCall, PyPlot, Seaborn

    function plotSIRPyPlot(t, SIR, alpha, beta, N, outputFileName="plot", Display=true, save=true)
        #=
        Inputs
        t              : Array of times at which events have occured
        SIR            : Array of arrays of Num people susceptible, infected and recovered at
                         each t
        N              : Population size
        alpha          : probability of infected person recovering [0,1]
        beta           : probability of susceptible person being infected [0,1]
        outputFileName : the name/location to save the plot as/in

        Outputs
        png            : plot of SIR model over time [by default]
        =#

        Seaborn.set()
        set_style("ticks")
        Seaborn.set_style("white")

        fig = plt.figure(dpi=300)
        plt.plot(t, SIR[1], label="Susceptible", lw = 2, figure=fig)
        plt.plot(t, SIR[2], label="Infected", lw = 2, figure=fig)
        plt.plot(t, SIR[3], label="Recovered", lw=2, figure=fig)
        try
            plt.plot(t, SIR[4], label="Deceased", lw=2, figure=fig)
        catch
        end
        plt.xlabel("Time")
        plt.ylabel("Population Number")
        plt.suptitle("SIR model over time with a population size of $N")
        plt.title("For alpha = $alpha and beta $beta")
        plt.legend()

        if Display
            # required to display graph on plots.
            display(fig)
        end
        if save
            # Save graph as pngW
            fig.savefig(outputFileName)

        end
        close()
    end

    function plotBranchPyPlot(t, SIR, N, outputFileName="plot", subtitle="", Display=true, save=true)
        #=
        Inputs
        t              : Array of times at which events have occured
        SIR            : Array of arrays of Num people susceptible, infected and recovered at
                         each t
        N              : Population size
        alpha          : probability of infected person recovering [0,1]
        beta           : probability of susceptible person being infected [0,1]
        outputFileName : the name/location to save the plot as/in

        Outputs
        png            : plot of SIR model over time [by default]
        =#

        Seaborn.set()
        set_style("ticks")
        Seaborn.set_style("white")

        fig = plt.figure(dpi=300)
        plt.plot(t, SIR[:,1], label="Susceptible", lw=2, figure=fig)
        plt.plot(t, SIR[:,2], label="Infected", lw=2, figure=fig)
        plt.plot(t, SIR[:,3], label="Recovered", lw=2, figure=fig)

        plt.xlabel("Time")
        plt.ylabel("Population Number")
        plt.suptitle("Branching SIR model over time with a population size of $N")
        plt.title("$subtitle")
        plt.legend()

        if Display
            # required to display graph on plots.
            display(fig)
        end
        if save
            # Save graph as pngW
            fig.savefig(outputFileName)

        end
        close()
    end

    function plotSimpleBranchPyPlot(t, SIR, N, outputFileName="plot", subtitle="", Display=true, save=true)
        #=
        Inputs
        t              : Array of times at which events have occured / generations
        SIR            : Array of arrays of Num people susceptible, infected and recovered at
                         each t
        N              : Population size
        alpha          : probability of infected person recovering [0,1]
        beta           : probability of susceptible person being infected [0,1]
        outputFileName : the name/location to save the plot as/in

        Outputs
        png            : plot of SIR model over time [by default]
        =#

        Seaborn.set()
        set_style("ticks")
        Seaborn.set_style("white")

        fig = plt.figure(dpi=300)
        plt.plot(t, SIR[:,1], "x", label="Susceptible", lw=2, figure=fig)
        plt.plot(t, SIR[:,2], "x", label="Infected", lw=2, figure=fig)
        plt.plot(t, SIR[:,3] ,"x", label="Recovered", lw=2, figure=fig)

        plt.xlabel("Time")
        plt.ylabel("Population Number")
        plt.suptitle("Branching SIR model over time with a population size of $N")
        plt.title("$subtitle")
        plt.legend()

        if Display
            # required to display graph on plots.
            display(fig)
        end
        if save
            # Save graph as pngW
            fig.savefig(outputFileName)

        end
        close()
    end
end  # module plots
