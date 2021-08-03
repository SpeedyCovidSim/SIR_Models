#=
This is a code base for comparing the discrete solution to a branching process
SIR simulation with the first reaction and next reaction solutions.
Much of this code has been copied over from ODEVerifySoln.jl

Author: Joel Trent

Values should be pretty close for all cases if the time_step is small enough
=#


module BranchVerifySoln

    using LightGraphs, PyPlot, Seaborn, Dierckx, StatsBase #, MetaGraphs

    export branchVerifyPlot, meanAbsError, initSIRArrays, multipleSIRMeans,
        multipleLinearSplines, branchTimeStepPlot

    function branchVerifyPlot(Smean, Imean, Rmean, discreteArray, times, title, outputFileName, First=true, Display=true, save=false, Discrete=true)

        #PyPlot.rcParams["figure.dpi"] = 300
        typeSim = ""
        if First
            typeSim = "First"
        else
            typeSim = "Next"
        end

        typeSim2 = ""
        if Discrete
            typeSim2 = "Discrete"
        else
            typeSim2 = "Simple BP"
        end

        Seaborn.set()
        Seaborn.set_color_codes("pastel")
        fig = plt.figure(dpi=300)
        plt.plot(times, Smean, "k-", label="S - $typeSim", lw=2.5, figure=fig)
        plt.plot(times, Imean, "b-", label="I - $typeSim", lw=2.5, figure=fig)
        plt.plot(times, Rmean, "r-", label="R - $typeSim", lw=2.5, figure=fig)

        plt.plot(times, discreteArray[:,1], "g-.", label="S - $typeSim2", lw=1.5, figure=fig, alpha = 1)
        plt.plot(times, discreteArray[:,2], "w-.", label="I - $typeSim2", lw=1.5, figure=fig, alpha = 1)
        plt.plot(times, discreteArray[:,3], "k-.", label="R - $typeSim2", lw=1.5, figure=fig, alpha = 1)


        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.suptitle("Branching Process Simulation")
        plt.title(title)
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

    function branchTimeStepPlot(Array1, Array2, Array3, times1, times2, times3, title, outputFileName, Display=true, save=false)

        Seaborn.set()
        Seaborn.set_color_codes("pastel")
        fig = plt.figure(dpi=300)

        plt.plot(times1, Array1[:,1], "g-", label="S - stepsize = $(times1[2]-times1[1])", lw=1.5, figure=fig, alpha = 0.7)
        plt.plot(times1, Array1[:,2], "k-", label="I - stepsize = $(times1[2]-times1[1])", lw=1.5, figure=fig, alpha = 0.7)
        plt.plot(times1, Array1[:,3], "r-", label="R - stepsize = $(times1[2]-times1[1])", lw=1.5, figure=fig, alpha = 0.7)

        plt.plot(times2, Array2[:,1], "g-.", label="S - stepsize = $(times2[2]-times2[1])", lw=1.5, figure=fig, alpha = 0.7)
        plt.plot(times2, Array2[:,2], "k-.", label="I - stepsize = $(times2[2]-times2[1])", lw=1.5, figure=fig, alpha = 0.7)
        plt.plot(times2, Array2[:,3], "r-.", label="R - stepsize = $(times2[2]-times2[1])", lw=1.5, figure=fig, alpha = 0.7)

        plt.plot(times3, Array3[:,1], "g:", label="S - stepsize = $(times3[2]-times3[1])", lw=1.5, figure=fig, alpha = 0.7)
        plt.plot(times3, Array3[:,2], "k:", label="I - stepsize = $(times3[2]-times3[1])", lw=1.5, figure=fig, alpha = 0.7)
        plt.plot(times3, Array3[:,3], "r:", label="R - stepsize = $(times3[2]-times3[1])", lw=1.5, figure=fig, alpha = 0.7)

        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.suptitle("Branching Process Timestep")
        plt.title(title)
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

    function meanAbsError(Smean, Imean, Rmean, DiscreteArray)
        misfitS = sum(abs.(Smean - DiscreteArray[:,1]))/length(Smean)
        misfitI = sum(abs.(Imean - DiscreteArray[:,2]))/length(Imean)
        misfitR = sum(abs.(Rmean - DiscreteArray[:,3]))/length(Rmean)

        return misfitS, misfitI, misfitR
    end

    function initSIRArrays(tspan, time_step, numSims)
        # Julia is column major
        StStep = zeros(Int(tspan[end]/time_step+1),numSims)
        ItStep = copy(StStep)
        RtStep = copy(StStep)

        return StStep, ItStep, RtStep
    end

    function multipleSIRMeans(StStep::Array{}, ItStep::Array{}, RtStep::Array{})
        Smean = mean(StStep, dims = 2)
        Imean = mean(ItStep, dims = 2)
        Rmean = mean(RtStep, dims = 2)
        return Smean, Imean, Rmean
    end

    function multipleLinearSplines(state_totals_all::Array{}, t::Array{}, times::Array{})
        splineS = Spline1D(t, state_totals_all[:,1], k=1)
        splineI = Spline1D(t, state_totals_all[:,2], k=1)
        splineR = Spline1D(t, state_totals_all[:,3], k=1)

        return splineS(times), splineI(times), splineR(times)
    end

end  # modulebranchVerify Soln
