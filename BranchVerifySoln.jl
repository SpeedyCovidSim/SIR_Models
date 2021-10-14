#=
This is a code base for comparing the discrete solution to a branching process
SIR simulation with the first reaction and next reaction solutions.
Much of this code has been copied over from ODEVerifySoln.jl

Author: Joel Trent

Values should be pretty close for all cases if the time_step is small enough
=#


module BranchVerifySoln

    using LightGraphs, PyPlot, Seaborn, Dierckx, StatsBase, Statistics #, MetaGraphs

    export branchVerifyPlot, meanAbsError, initSIRArrays, multipleSIRMeans,
        multipleLinearSplines, branchTimeStepPlot, branchSideBySideVerifyPlot,
        branchSideBySideVerifyPlot2, branch2wayVerifyPlot, Dots, Lines, singleSpline,
        singleLinearSpline

    struct Dots; end
    struct Lines; end

    function branchVerifyPlot(Smean::Array{Float64,2}, Imean::Array{Float64,2},
        Rmean::Array{Float64,2}, discreteArray::Array{Float64,2}, times::Array{Float64,1}, title::String,
        outputFileName::String, First=true, Display=true, save=false, Discrete=true, Tree=true)

        #PyPlot.rcParams["figure.dpi"] = 300
        typeSim = ""
        if First
            typeSim = "FRBPM"
        else
            typeSim = "NRBPM"
        end

        typeSim2 = ""
        if Discrete
            typeSim2 = "DBPM"
        elseif Tree
            typeSim2 = "TTBPM"
        else
            typeSim2 = "STBPM"
        end

        Seaborn.set()
        set_style("ticks")
        Seaborn.set_color_codes("pastel")
        fig = plt.figure(dpi=300)
        plt.plot(times, Smean, "k-", label="S - $typeSim", lw=2.5, figure=fig)
        plt.plot(times, Imean, "b-", label="I - $typeSim", lw=2.5, figure=fig)
        plt.plot(times, Rmean, "r-", label="R - $typeSim", lw=2.5, figure=fig)

        plt.plot(times, discreteArray[:,1], "g-.", label="S - $typeSim2", lw=1.5, figure=fig, alpha = 1)
        plt.plot(times, discreteArray[:,2], color="tab:gray", linestyle="-.", label="I - $typeSim2", lw=1.5, figure=fig, alpha = 1)
        plt.plot(times, discreteArray[:,3], "k-.", label="R - $typeSim2", lw=1.5, figure=fig, alpha = 1)


        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.suptitle("Branching Process Simulation")
        plt.title(title)
        # plt.rcParams["font.family"] = "DejaVu Sans Mono"
        plt.legend(prop=plt.matplotlib.font_manager.FontProperties(family="DejaVu Sans Mono"))

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

    function branchVerifyPlot(meanDetect::Array{Float64,1}, meanDetect_disc::Array{Float64,1},
        meanDetect_disc_match::Array{Float64,1}, times::Array{Float64,1}, title::String,
        outputFileName::String, ChangeDisc=true, Display=true, save=false, lowR=false)

        meanDetectDaily = diff(vcat([0],meanDetect))
        meanDetectDaily_disc = diff(vcat([0],meanDetect_disc))
        meanDetectDaily_disc_match = diff(vcat([0],meanDetect_disc_match))
        # meanDetectDaily = meanDetect
        # meanDetectDaily_disc = meanDetect_disc

        #PyPlot.rcParams["figure.dpi"] = 300

        typeSim = "NRBPM"

        typeSim2 = "DBPM"


        Seaborn.set()
        set_style("ticks")
        Seaborn.set_color_codes("pastel")
        fig = plt.figure(dpi=300)

        if ChangeDisc
            plt.plot(times, meanDetectDaily, "k-", label="$typeSim", lw=2.5, figure=fig)

            plt.plot(times, meanDetectDaily_disc, "b-.", label="$typeSim2, identical model", lw=1.5, figure=fig, alpha = 1)
            plt.plot(times, meanDetectDaily_disc_match, "r-.", label="$typeSim2, higher Reff model", lw=1.5, figure=fig, alpha = 1)
        else
            plt.plot(times, meanDetectDaily, "b-", label="$typeSim2", lw=2.5, figure=fig)

            plt.plot(times, meanDetectDaily_disc, color="tab:gray", linestyle="-.", label="$typeSim, lower seed cases", lw=1.5, figure=fig, alpha = 1)
            plt.plot(times, meanDetectDaily_disc_match, "k-.", label="$typeSim, lower seed cases + higher Reff", lw=1.5, figure=fig, alpha = 1)
        end

        plt.xlabel("Time")
        plt.ylabel("Number of Detected Cases per Day")
        plt.suptitle("Branching Process Simulation")
        plt.title(title)
        plt.legend(loc = "upper right", prop=plt.matplotlib.font_manager.FontProperties(family="DejaVu Sans Mono", size=9))

        if lowR
            plt.ylim([0,20])
        else
            plt.ylim([0,75])
        end

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

    function branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName,
        allRealisations=true, Display=true, save=false, alphaMultiplier=1.0)
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

        if allRealisations
            for i in 1:length(x1[1,:])
                if i == 1
                    ax[1].plot(times, x1[:,i], "b-", label="Homogeneous Realisation", lw=2, alpha = 0.2*alphaMultiplier)
                else
                    ax[1].plot(times, x1[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
                end
            end

            for i in 1:length(x2[1,:])
                if i == 1
                    ax[2].plot(times, x2[:,i], "b-", label="Heterogeneous Realisation", lw=2, alpha = 0.2*alphaMultiplier)
                else
                    ax[2].plot(times, x2[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
                end
            end

            labelx1 = ["Homogeneous Mean" for i in 1:length(timesVector)]
            labelx2 = ["Heterogeneous Mean" for i in 1:length(timesVector)]
            Seaborn.lineplot(x = timesVector, y = [x1...], hue = labelx1, palette = "flare", ci=nothing, ax=ax[1])
            Seaborn.lineplot(x = timesVector, y = [x2...], hue = labelx2, palette = "flare", ci=nothing, ax=ax[2])
        else
            labelx1 = ["Homogeneous Mean and SD" for i in 1:length(timesVector)]
            labelx2 = ["Heterogeneous Mean and SD" for i in 1:length(timesVector)]
            Seaborn.lineplot(x = timesVector, y = [x1...], hue = labelx1, palette = "flare", ci="sd", ax=ax[1])
            Seaborn.lineplot(x = timesVector, y = [x2...], hue = labelx2, palette = "flare", ci="sd", ax=ax[2])
        end

        # meanx1 = mean(x1, dims=2)
        # meanx2 = mean(x2, dims=2)
        #
        # ax[1].plot(times, meanx1, "r-", label="Homogeneous Mean", lw=2.5, alpha = 1.0)
        # ax[2].plot(times, meanx2, "r-", label="Heterogeneous Mean", lw=2.5, alpha = 1.0)

        maxy = max(maximum(x1), maximum(x2))

        ax[1].set_ylim([0, maxy])
        ax[1].legend(loc = "upper left")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Cumulative Number of Infections")
        ax[1].set_title("Homogeneous Reproduction Number")

        ax[2].set_ylim([0, maxy])
        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("Cumulative Number of Infections")
        ax[2].set_title("Heterogeneous Reproduction Number")
        ax[2].legend(loc = "upper left")

        plt.suptitle(title,figure=f)
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

    function branchSideBySideVerifyPlot2(x11, x12, x21, x22, times, title, outputFileName,
        allRealisations=true, Display=true, save=false, alphaMultiplier=1.0)
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
        for i in 1:(length(x11[1,:]))
            timesVector = vcat(timesVector, times)
        end

        if allRealisations
            for i in 1:length(x11[1,:])
                if i == 1
                    ax[1].plot(times, x11[:,i], "b-", label="Homogeneous Realisation", lw=2, alpha = 0.2*alphaMultiplier)
                    ax[1].plot(times, x12[:,i], "g-", label="Homogeneous Realisation - Isolation", lw=2, alpha = 0.2*alphaMultiplier)
                else
                    ax[1].plot(times, x11[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
                    ax[1].plot(times, x12[:,i], "g-", lw=2, alpha = 0.09*alphaMultiplier)
                end
            end

            for i in 1:length(x21[1,:])
                if i == 1
                    ax[2].plot(times, x21[:,i], "b-", label="Heterogeneous Realisation", lw=2, alpha = 0.2*alphaMultiplier)
                    ax[2].plot(times, x22[:,i], "g-", label="Heterogeneous Realisation - Isolation", lw=2, alpha = 0.2*alphaMultiplier)
                else
                    ax[2].plot(times, x21[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
                    ax[2].plot(times, x22[:,i], "g-", lw=2, alpha = 0.09*alphaMultiplier)
                end
            end

            labelx11 = ["Homogeneous Mean" for i in 1:length(timesVector)]
            labelx21 = ["Heterogeneous Mean" for i in 1:length(timesVector)]
            labelx12 = ["Homogeneous Mean - Isolation" for i in 1:length(timesVector)]
            labelx22 = ["Heterogeneous Mean - Isolation" for i in 1:length(timesVector)]

            x1cat = vcat([x11...],[x12...])
            labelx1 = vcat(labelx11, labelx12)
            x2cat = vcat([x21...],[x22...])
            labelx2 = vcat(labelx21, labelx22)

            # println(length(x1cat))
            # println(length(labelx1))
            # println(length(timesVector))

            timesVector = vcat(timesVector, timesVector)
            Seaborn.lineplot(x = timesVector, y = x1cat, hue = labelx1, palette = "flare", ci=nothing, ax=ax[1])
            Seaborn.lineplot(x = timesVector, y = x2cat, hue = labelx2, palette = "flare", ci=nothing, ax=ax[2])

        else
            labelx11 = ["Homogeneous Mean and SD" for i in 1:length(timesVector)]
            labelx21 = ["Heterogeneous Mean and SD" for i in 1:length(timesVector)]
            labelx12 = ["Homogeneous Mean and SD - Isolation" for i in 1:length(timesVector)]
            labelx22 = ["Heterogeneous Mean and SD - Isolation" for i in 1:length(timesVector)]

            x1cat = vcat([x11...],[x12...])
            labelx1 = vcat(labelx11, labelx12)
            x2cat = vcat([x21...],[x22...])
            labelx2 = vcat(labelx21, labelx22)

            timesVector = vcat(timesVector, timesVector)
            Seaborn.lineplot(x = timesVector, y = x1cat, hue = labelx1, palette = "flare", ci="sd", ax=ax[1])
            Seaborn.lineplot(x = timesVector, y = x2cat, hue = labelx2, palette = "flare", ci="sd", ax=ax[2])
        end

        # meanx1 = mean(x1, dims=2)
        # meanx2 = mean(x2, dims=2)
        #
        # ax[1].plot(times, meanx1, "r-", label="Homogeneous Mean", lw=2.5, alpha = 1.0)
        # ax[2].plot(times, meanx2, "r-", label="Heterogeneous Mean", lw=2.5, alpha = 1.0)

        maxy = max(maximum(x11), maximum(x21), maximum(x12), maximum(x22))

        ax[1].set_ylim([0, maxy])
        ax[1].legend(loc = "upper left")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Cumulative Number of Infections")
        ax[1].set_title("Homogeneous Reproduction Number")

        ax[2].set_ylim([0, maxy])
        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("Cumulative Number of Infections")
        ax[2].set_title("Heterogeneous Reproduction Number")
        ax[2].legend(loc = "upper left")

        plt.suptitle(title,figure=f)
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

    function branch2wayVerifyPlot(x1, x2, times, title, outputFileName, dotsOrLines::Union{Dots,Lines}, Display=true, save=false)

        #PyPlot.rcParams["figure.dpi"] = 300
        x1PlotType = ""
        x2PlotType = ""

        if dotsOrLines == Dots()
            x1PlotType = "x"
            x2PlotType = "*"
        elseif dotsOrLines == Lines()
            x1PlotType = "-"
            x2PlotType = "-."
        else
            @warn No line type specified. Using lines.
            x1PlotType = "-"
            x2PlotType = "-."
        end

        Seaborn.set()
        set_style("ticks")
        Seaborn.set_color_codes("pastel")
        fig = plt.figure(dpi=300)

        plt.plot(times, x1, "b$x1PlotType", label="I - Basic BP", lw=2.5, figure=fig)
        plt.plot(times, x2, "r$x2PlotType", label="I - Geometric Series", lw=1.5, figure=fig, alpha = 1)

        plt.xlabel("Time")
        plt.ylabel("Cumulative Number of Infections")
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
        set_style("ticks")
        Seaborn.set_color_codes("pastel")
        fig = plt.figure(dpi=300)

        plt.plot(times1, Array1[:,1], "k-", label="S, Δt=$(times1[2]-times1[1])", lw=1.5, figure=fig, alpha=0.7)
        plt.plot(times1, Array1[:,2], "b-", label="I, Δt=$(times1[2]-times1[1])", lw=1.5, figure=fig, alpha=0.7)
        plt.plot(times1, Array1[:,3], "r-", label="R, Δt=$(times1[2]-times1[1])", lw=1.5, figure=fig, alpha=0.7)

        plt.plot(times2, Array2[:,1], "k-.", label="S, Δt=$(times2[2]-times2[1])", lw=1.5, figure=fig, alpha=0.7)
        plt.plot(times2, Array2[:,2], "b-.", label="I, Δt=$(times2[2]-times2[1])", lw=1.5, figure=fig, alpha=0.7)
        plt.plot(times2, Array2[:,3], "r-.", label="R, Δt=$(times2[2]-times2[1])", lw=1.5, figure=fig, alpha=0.7)

        plt.plot(times3, Array3[:,1], "k:", label="S, Δt=$(times3[2]-times3[1])", lw=1.5, figure=fig, alpha=0.7)
        plt.plot(times3, Array3[:,2], "b:", label="I, Δt=$(times3[2]-times3[1])", lw=1.5, figure=fig, alpha=0.7)
        plt.plot(times3, Array3[:,3], "r:", label="R, Δt=$(times3[2]-times3[1])", lw=1.5, figure=fig, alpha=0.7)

        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.suptitle("Branching Process Timestep")
        plt.title(title)
        plt.legend(prop=plt.matplotlib.font_manager.FontProperties(family="DejaVu Sans Mono", size=9))


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
        splineS = Spline1D(t, state_totals_all[:,1], k=1, bc="nearest")
        splineI = Spline1D(t, state_totals_all[:,2], k=1, bc="nearest")
        splineR = Spline1D(t, state_totals_all[:,3], k=1, bc="nearest")

        return splineS(times), splineI(times), splineR(times)
    end

    function singleSpline(x::Array{}, t::Array, times::Array)
        spline = Spline1D(t, x, k=2)

        return spline(times)
    end

    function singleLinearSpline(x::Array{}, t::Array, times::Array)
        spline = Spline1D(t, x, k=1, bc="nearest")

        return spline(times)
    end
end  # modulebranchVerify Soln
