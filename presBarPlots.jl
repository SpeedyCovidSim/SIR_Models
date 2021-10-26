#=
The script used for generating a simple set of plots used in the project presentation. 

Author: Joel Trent
=#

using PyPlot, Seaborn

function simpleBarPlot(yVals, xVals, outputFileName, Display, save)
    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    # Seaborn.set_color_codes("Set2")
    # Seaborn.set_style("white")

    fig = plt.figure(figsize=[6,4],dpi=300)

    Seaborn.barplot(x=xVals, y=yVals, palette="Set2", orient="h")
    plt.ylabel("Algorithm")
    plt.title("Relative Median Performance (Lower Is Better)")
    plt.tight_layout()

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

# Julia v Py ####################################################
outputFileName = "./juliaGraphs/presentationGraphs/JuliaVPy"
simpleBarPlot(["Julia", "Python"], [1,70], outputFileName, true, true)

# Network models ####################################################
outputFileName = "./juliaGraphs/presentationGraphs/DirectVNextVFirst"
simpleBarPlot(["Direct", "Next \nReaction", "First \nReaction"], [1,0.8,10], outputFileName, true, true)

# Network models ####################################################
outputFileName = "./juliaGraphs/presentationGraphs/DirectVNext"
simpleBarPlot(["Direct", "Next \nReaction"], [38,1], outputFileName, true, true)

# BP models ####################################################
outputFileName = "./juliaGraphs/presentationGraphs/BP_NextVFirstVDisc"
simpleBarPlot(["Discrete \nΔt=1 day", "Next \nReaction", "Discrete \nΔt=0.02 days", "First \nReaction"], [1,0.8,40,100], outputFileName, true, true)

outputFileName = "./juliaGraphs/presentationGraphs/BP_NextVDisc"
simpleBarPlot(["Discrete \nΔt=1 day", "Next \nReaction", "Discrete \nΔt=0.02 days"], [1,0.8,40], outputFileName, true, true)

outputFileName = "./juliaGraphs/presentationGraphs/BP_NextVThinned"
simpleBarPlot(["Thinned \nTree", "Next \nReaction"], [5,1], outputFileName, true, true)
