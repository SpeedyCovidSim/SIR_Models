#=
This is a code base for the plotting function of a simple SIR simulation
using the Gillespie Direct Method. Using PyPlot

Author: Joel Trent and Josh Looker
=#


module plotsPyPlot

    using PyCall, PyPlot, Seaborn, DataFrames, StatsBase

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

    function plotCumulativeInfections(x1, times, timesOfInterest, title, outputFileName,
        Display=true, save=false, alphaMultiplier=1.0)
        #=
        Plot multiple realisations of x1 and x2 as well as their means.
        =#

        Seaborn.set()
        set_style("ticks")
        set_color_codes("pastel")
        # fig = plt.figure(dpi=300)

        # Initialise plots - need figure size to make them square and nice
        f,ax = Seaborn.subplots(1,2, figsize=(10,4), dpi=300)

        timesVector = []
        for i in 1:length(x1[1,:])
            timesVector = vcat(timesVector, times)
        end

        for i in 1:length(x1[1,:])
            if i == 1
                ax[1].plot(times, x1[:,i], "b-", label="Realisations", lw=2, alpha = 0.2*alphaMultiplier)
            else
                ax[1].plot(times, x1[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
            end
        end

        labelx1 = ["Mean and SD" for i in 1:length(timesVector)]

        Seaborn.lineplot(x = timesVector, y = [x1...], hue = labelx1, palette = "flare", ci="sd", ax=ax[1])

        BP_df = DataFrame()
        BP_df.timesVector = timesVector
        BP_df.infections = [x1...]
        BP_df.label = labelx1

        filtered_BP = filter(row -> row.timesVector in timesOfInterest, BP_df, view=false)

        Seaborn.violinplot(x=filtered_BP.timesVector, y=filtered_BP.infections, hue=filtered_BP.timesVector,
            bw=0.4, cut=0, scale="count",palette = "Set2",ax=ax[2])


        for i in timesOfInterest
            # meanNumber = round(mean(filter(row -> row.timesVector == i, filtered_BP).infections))
            # println("Mean number of infections is: $meanNumber")
            println("Summary Statistics at Day #$i")
            describe(filter(row -> row.timesVector == i, filtered_BP).infections)
        end

        # maxy = maximum(x1)
        maxy=1000

        ax[1].set_ylim([0, maxy*0.6])
        ax[1].legend(loc = "upper left")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Cumulative Number of Infections")
        ax[1].set_title("Cumulative Infections with one seed case")

        ax[2].set_ylim([0, maxy*0.6])
        ax[2].set_xlabel("Time")
        ax[2].legend(loc = "upper left")
        ax[2].set_ylabel("Cumulative Number of infections")
        ax[2].set_title("Infection Count At Times of interest")


        plt.suptitle(title)
        # plt.title(title)


        # Dodge the other plots
        # plt.tight_layout(pad = 0.8, h_pad=0.01, w_pad=0.01)
        plt.tight_layout(h_pad=0.01)
        # despine()

        if Display
            # required to display graph on plots.
            display(f)
        end
        if save
            # Save graph as pngW
            f.savefig(outputFileName)

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
