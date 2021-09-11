using BenchmarkTools
using DataFrames
using Distributions, Random, StatsBase, Statistics
using LightGraphs, GraphPlot, NetworkLayout
using PyPlot, Seaborn
using ProgressMeter
using CSV
using Dates
using PlotlyJS
using PyCall

# import required modules
push!( LOAD_PATH, "./" )
using plotsPyPlot: plotBranchPyPlot, plotSimpleBranchPyPlot, plotCumulativeInfections,
    plotBenchmarksViolin
using BranchVerifySoln
using branchingProcesses
using BPVerifySolutions
using outbreakPostProcessing

# set some globals
global const PROGRESS__METER__DT = 0.2
global const DAY__ZERO__DATE = Date("17-08-2021", dateformat"d-m-y")
global ALERT__REFF__MAX__VAL = 1.3
global USING__ALERT__MAX__VAL = true
global CHANGING__ALERT__LEVELS = false
global ALERT__CHANGE__DATE = Date("16-09-2021", dateformat"d-m-y")

# import Ccandu conditioning
py"""
import sys
sys.path.append("/Users/joeltrent/Documents/GitHub/SIR_Models")
from transpose import transposeCSV
from SeabornDistplot import seabornDist, plotReffHist
"""
# conditionEnsemble = py"condition"
transposeCSV = py"transposeCSV"
plotReffAL3 = py"seabornDist"
plotReffHist = py"plotReffHist"

function daysSinceZero()
    return Dates.value(Date(now()) - DAY__ZERO__DATE)
end

function daysFromZeroToALChange()
    return Dates.value(ALERT__CHANGE__DATE - DAY__ZERO__DATE)
end

function Df_transpose(df)
    #=
    Given a DataFrame, return it's transpose
    =#
    return DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])
end

function convertCSVtoConditioning(origPath, newPath, alreadyTransposed=false)

    case_df = DataFrame()

    if !alreadyTransposed
        case_df = DataFrame(CSV.File(origPath))
        # case_df.time = convert.(Int64, case_df.time)
        CSV.write(newPath, case_df)

        transposeCSV(newPath, newPath)
    else
        case_df = DataFrame(CSV.File(origPath,header=2))
        # case_df = select(case_df, 2:ncol(case_df))
        CSV.write(newPath, case_df)
    end

    case_df = DataFrame(CSV.File(newPath))

    rename!(case_df,"time" => "metric")

    runColumn = ["" for i in 1:nrow(case_df)]

    cumulative = true
    type = ""
    if cumulative
        type = "Cumulative"
    else
        type = "Daily"
    end

    j = 1
    for i in 1:length(runColumn)
        if rem(i,2) == 1
            runColumn[i] = "$j Known"
            case_df[i, :metric] = "Known Cases"
        else
            runColumn[i] = "$j $type"
            case_df[i, :metric] = "$type Cases"
            j+=1
        end
    end

    case_df = insertcols!(case_df, 1, :run=>runColumn)

    # println(case_df.run)
    CSV.write(newPath, case_df)

    return nothing
end
#
# origPath = "/Users/joeltrent/Documents/GitHub/SIR_Models/August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_MatchMike.csv"
# newPath = "/Users/joeltrent/Documents/GitHub/SIR_Models/August2020OutbreakFit/CSVOutputs/config_1_MatchMike.timeseries.csv"
# convertCSVtoConditioning(origPath, newPath);

# origPath = "/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_10Sep_contact.csv";
# newPath = "/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/config_10_created10Sep.timeseries.csv";
# convertCSVtoConditioning(origPath, newPath);

# origPath = "/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_created5Sep_8Sep.csv";
# newPath = "/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/config_8_created5Sep.timeseries.csv";
# convertCSVtoConditioning(origPath, newPath);

function observedCasesAugust2020(daily::Bool)
    observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
    122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
    184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]

    if daily
        observedIDetect = diff(vcat([0],observedIDetect))
    end
    tDetect = collect(0:length(observedIDetect)-1)

    return observedIDetect, tDetect
end

function observedCases()
    # observedIDetect = cumsum([1,4,4,15,20,26,38,36,56,67,86,75,81])
    # observedIDetect = cumsum([5,4,16,20,27,36,37,60,69,79,76,79,54,65,59,45])
    observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566, 615, 690, 739, 767, 787, 807, 827, 848, 863, 876]

    tDetect = collect(0:length(observedIDetect)-1)
    return observedIDetect, tDetect
end

function BPbenchmarking(numSimsScaling::Int64, benchmarkRange)

    println("Benchmark #1: Discrete (1 day and 0.02 day timesteps) vs Next and First React Style, Speed, Varied Max Case Size")
    if 1 in benchmarkRange

        # initialise variables
        maxCases = [1,2,5,10,25,50,75]*10^2#,500,1000]*10^2
        tspan = (0.0,100.0)

        it_count = 10000*10^2 ./ maxCases
        # [2000,1000,200,100,40,20,5]
        it_count = convert.(Int, round.(it_count ./ numSimsScaling))
        # it_count_first = [2000, 1000, 800, 200, 50, 20, 4, 2]

        time_df = DataFrame(time=Float64[], type=String[], population=Int64[])

        # setup a loop that benchmarks each function for different N values
        for i::Int64 in 1:length(maxCases)

            timesteps = [0.02,1]

            @showprogress PROGRESS__METER__DT for j in 1:(it_count[i])

                model = init_model_pars(tspan[1], tspan[end], 5*10^6, maxCases[i], [5*10^6-10,10,0]);
                model.stochasticRi = true
                model.reproduction_number = 3
                model.p_test = 0.4

                # model = init_model_pars(tspan[1], tspan[end], 5*10^3, maxCases, [5*10^3-10,10,0])
                models = [deepcopy(model) for i in 1:4]

                population_df = initDataframe(models[1]);
                population_dfs = [deepcopy(population_df) for i in 1:4]

                k = 1
                if rem(j, 6) == 0
                    time_dir = @elapsed discrete_branch!(population_dfs[k], models[k], timesteps[1])
                    push!(time_df, [log10(time_dir), "Discrete, Timestep=$(timesteps[1])", maxCases[i]])
                end

                k+=1
                time_dir = @elapsed discrete_branch!(population_dfs[k], models[k], timesteps[2])
                push!(time_df, [log10(time_dir), "Discrete, Timestep=$(timesteps[2])", maxCases[i]])

                k+=1
                if rem(j, 10) == 0
                    time_dir = @elapsed firstReact_branch!(population_dfs[k], models[k])
                    push!(time_df, [log10(time_dir), "First React Style", maxCases[i]])
                end

                k+=1
                time_dir = @elapsed nextReact_branch!(population_dfs[k], models[k])
                push!(time_df, [log10(time_dir), "Next React Style", maxCases[i]])

            end

            println("Completed iteration #$i")

        end

        outputFileName = "Benchmarks/SimulationTimesMikeBranch"
        xlabel = "Maximum Case Size"
        plotBenchmarksViolin(time_df.population, time_df.time, time_df.type, outputFileName,
        xlabel, true, true)
    end

    println("Benchmark #2: Discrete (1 day and 0.02 day timesteps) vs Next React Style, Speed, Varied Max Case Size")
    if 2 in benchmarkRange

        # initialise variables
        maxCases = [25,50,75,500,1000]*10^2
        tspan = (0.0,100.0)

        it_count = 20000*10^2 ./ maxCases
        # [2000,1000,200,100,40,20,5]
        it_count = convert.(Int, round.(it_count ./ numSimsScaling))
        # it_count_first = [2000, 1000, 800, 200, 50, 20, 4, 2]

        time_df = DataFrame(time=Float64[], type=String[], population=Int64[])

        # setup a loop that benchmarks each function for different N values
        for i::Int64 in 1:length(maxCases)

            timesteps = [0.02,1]

            @showprogress PROGRESS__METER__DT for j in 1:(it_count[i])

                model = init_model_pars(tspan[1], tspan[end], 5*10^6, maxCases[i], [5*10^6-10,10,0]);
                model.stochasticRi = true
                model.reproduction_number = 3
                model.p_test = 0.4

                # model = init_model_pars(tspan[1], tspan[end], 5*10^3, maxCases, [5*10^3-10,10,0])
                models = [deepcopy(model) for i in 1:4]

                population_df = initDataframe(models[1]);
                population_dfs = [deepcopy(population_df) for i in 1:4]

                k = 1
                if rem(j, 5) == 0
                    time_dir = @elapsed discrete_branch!(population_dfs[k], models[k], timesteps[1])
                    push!(time_df, [log10(time_dir), "Discrete, Timestep=$(timesteps[1])", maxCases[i]])
                end

                k+=1
                time_dir = @elapsed discrete_branch!(population_dfs[k], models[k], timesteps[2])
                push!(time_df, [log10(time_dir), "Discrete, Timestep=$(timesteps[2])", maxCases[i]])

                # k+=1
                # if rem(j, 10) == 0
                #     time_dir = @elapsed firstReact_branch!(population_dfs[k], models[k])
                #     push!(time_df, [log10(time_dir), "First React Style", maxCases[i]])
                # end

                k+=1
                time_dir = @elapsed nextReact_branch!(population_dfs[k], models[k])
                push!(time_df, [log10(time_dir), "Next React Style", maxCases[i]])

            end

            println("Completed iteration #$i")

        end

        outputFileName = "Benchmarks/SimulationTimesMikeBranch_noFirst"
        xlabel = "Maximum Case Size"
        plotBenchmarksViolin(time_df.population, time_df.time, time_df.type, outputFileName,
        xlabel, true, true)

        # Seaborn.set()
        # Seaborn.set_style("ticks")
        # fig = plt.figure(dpi=300)
        # # plt.violinplot(data)
        # Seaborn.violinplot(x=time_df.population, y=time_df.time, hue=time_df.type,
        #     bw=1.5, cut=0,scale="count",palette = "Set2" )
        #
        # plt.xlabel("Maximum Case Size")
        # plt.ylabel("Log10 Simulation time (log10(s))")
        # plt.title("Time To Complete Simulation")
        # # plt.title("For alpha = $alpha and beta $beta")
        # plt.legend(loc = "upper left")
        # display(fig)
        # fig.savefig("Benchmarks/SimulationTimesMikeBranch_noFirst")
        # close()
    end
end

function plotDailyCasesOutbreak(dailyConfirmedCases, dailyTimes, actualDailyCumCases,
    timesActual, title, outputFileName, two21::Bool=true, Display=true, save=false, conditioned=false, useDates=false)

    actualDailyCases = diff(vcat([0],actualDailyCumCases))
    # dailyTimes=timesDaily

    # only plot x number of days
    if two21 && false
        finalIndex = findfirst(dailyTimes.==15)

        dailyConfirmedCases = dailyConfirmedCases[1:finalIndex, :]
        dailyTimes = dailyTimes[1:finalIndex]
        actualDailyCases = actualDailyCases[1:min(finalIndex, length(actualDailyCases))]
    end

    # set up figure
    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(figsize=(8,6),dpi=300)

    # If using conditioned model ensemble, plot 50% (IQR) quantiles as well ----
    if conditioned

        quantiles95_1 = quantile2D(dailyConfirmedCases, 0.025)
        quantiles95_2 = quantile2D(dailyConfirmedCases, 0.975)
        quantiles50_1 = quantile2D(dailyConfirmedCases, 0.25)
        quantiles50_2 = quantile2D(dailyConfirmedCases, 0.75)

        # plotting 95% bands ---------------------------------------------------
        plt.plot(dailyTimes, quantiles95_1, "k--", label="95% Quantile Bands ", lw=2, alpha = 0.5)
        plt.plot(dailyTimes, quantiles95_2, "k--", lw=2, alpha = 0.5)

        plt.fill_between(dailyTimes, quantiles95_1, quantiles50_1, alpha=0.5, color = "r")
        plt.fill_between(dailyTimes, quantiles95_2, quantiles50_2, alpha=0.5, color = "r")

        # plotting 50% bands ---------------------------------------------------
        plt.plot(dailyTimes, quantiles50_1, "r--", label="50% Quantile Bands", lw=2, alpha = 0.8)
        plt.plot(dailyTimes, quantiles50_2, "r--", lw=2, alpha = 0.8)
        plt.fill_between(dailyTimes, quantiles50_1, quantiles50_2, alpha=0.5, color = "b")

        if two21
            # plt.ylim([0,120])
        end
    else
        quantiles1 = quantile2D(dailyConfirmedCases, 0.025)
        quantiles3 = quantile2D(dailyConfirmedCases, 0.975)
        plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.5, color = "r", label="95% Quantile Band")
    end


    # plot median daily detected cases and actual cases ------------------------
    plt.plot(dailyTimes, median(dailyConfirmedCases,dims=2), "r-", label="Median Daily Detected Cases", lw=2.5, figure=fig)

    # if in 2021, plot August 2020 data as well
    if two21
        observedIDetect, tDetect = observedCasesAugust2020(true)
        plt.plot(tDetect, observedIDetect, color="tab:gray", linestyle="-", label="August 2020 Daily Detected Cases", lw=2.5, figure=fig)
    end
    plt.plot(timesActual, actualDailyCases, "k-", label="Actual Daily Detected Cases", lw=2.5, figure=fig)


    if useDates
        plt.xticks(collect(dailyTimes[1]:10:dailyTimes[end]),
            [Dates.format(DAY__ZERO__DATE + Dates.Day(i*10), "u d") for i in 0:(length(collect(dailyTimes[1]:10:dailyTimes[end]))-1)])
        plt.xlabel("Date")

    else
        plt.xlabel("Days since detection")
    end
    plt.ylabel("Daily Detected Cases")
    # plt.suptitle("Branching Process Simulation")
    plt.title(title)
    plt.legend(loc = "upper right")
    # plt.ylim([0,150])

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

function plotDailyCasesMultOutbreak(dailyConfirmedCasesHigh, dailyConfirmedCasesMod,
    dailyConfirmedCasesLow, timesDaily, actualDailyCumCases, timesActual,
    title, outputFileName, two21::Bool=true, Display=true, save=false, useDates=true)

    actualDailyCases = diff(vcat([0],actualDailyCumCases))

    dailyTimes=timesDaily


    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(dpi=300)

    # High ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesHigh, 0.025)
    quantiles3 = quantile2D(dailyConfirmedCasesHigh, 0.975)
    plt.plot(dailyTimes, median(dailyConfirmedCasesHigh,dims=2), "b-", label="Median Daily Detected Cases - High", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.3, color = "b")
    ######################################################################

    # Mod ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesMod, 0.025)
    quantiles3 = quantile2D(dailyConfirmedCasesMod, 0.975)
    plt.plot(dailyTimes, median(dailyConfirmedCasesMod,dims=2), "r-", label="Median Daily Detected Cases - Moderate", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.3, color = "r")
    ######################################################################

    # Low ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesLow, 0.025)
    quantiles3 = quantile2D(dailyConfirmedCasesLow, 0.975)
    plt.plot(dailyTimes, median(dailyConfirmedCasesLow,dims=2), "g-", label="Median Daily Detected Cases - Low", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.3, color = "g")
    ######################################################################

    plt.plot(timesActual, actualDailyCases, "k-", label="Actual Daily Detected Cases", lw=2.5, figure=fig)

    # plt.plot(times, x2, "r$x2PlotType", label="I - Geometric Series", lw=1.5, figure=fig, alpha = 1)

    if two21
        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
        122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
        184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]

        observedIDetect = diff(vcat([0],observedIDetect))
        tDetect = collect(0:length(observedIDetect)-1)

        plt.plot(tDetect, observedIDetect, color="tab:gray", linestyle="-", label="August 2020 Daily Detected Cases", lw=2.5, figure=fig)
    end

    if two21 && false
        plt.xticks([-3,0,3,6,9,12,15],vcat(["Aug $(14+3*i)" for i in 0:5], ["Sep 01"]))
        plt.xlabel("Date")
    else
        plt.xlabel("Days since Detection")
    end
    plt.ylabel("Daily Detected Cases")
    # plt.suptitle("Branching Process Simulation")
    plt.title(title)
    plt.legend(loc = "upper right")

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

function plotDailyCasesMultOutbreak(dailyConfirmedCasesHigh, dailyConfirmedCasesMod,
    dailyConfirmedCasesLow, dailyConfirmedCasesBase, timesDaily, actualDailyCumCases,
    timesActual, title, outputFileName, two21::Bool=true, Display=true, save=false, useDates=true)

    actualDailyCases = diff(vcat([0],actualDailyCumCases))

    dailyTimes=timesDaily


    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(dpi=300)

    # High ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesHigh, 0.25)
    quantiles3 = quantile2D(dailyConfirmedCasesHigh, 0.75)
    plt.plot(dailyTimes, median(dailyConfirmedCasesHigh,dims=2), "b-", label="Median - High Increase (3x)", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.2, color = "b")
    ######################################################################

    # Mod ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesMod, 0.25)
    quantiles3 = quantile2D(dailyConfirmedCasesMod, 0.75)
    plt.plot(dailyTimes, median(dailyConfirmedCasesMod,dims=2), "r-", label="Median - Moderate Increase (2x)", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.2, color = "r")
    ######################################################################

    # Low ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesLow, 0.25)
    quantiles3 = quantile2D(dailyConfirmedCasesLow, 0.75)
    plt.plot(dailyTimes, median(dailyConfirmedCasesLow,dims=2), "g-", label="Median - Low Increase (1.5x)", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.2, color = "g")
    ######################################################################

    # Base ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesBase, 0.25)
    quantiles3 = quantile2D(dailyConfirmedCasesBase, 0.75)
    plt.plot(dailyTimes, median(dailyConfirmedCasesBase,dims=2), "c-", label="Median - No Change", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.2, color = "c")
    ######################################################################

    plt.plot(timesActual, actualDailyCases, "k-", label="Actual Daily Detected Cases", lw=2.5, figure=fig)

    # plt.plot(times, x2, "r$x2PlotType", label="I - Geometric Series", lw=1.5, figure=fig, alpha = 1)

    # plot date of AL change from AL4 to AL3
    if CHANGING__ALERT__LEVELS
        t_change = daysFromZeroToALChange()

        plt.plot([t_change, t_change], [0, maximum(quantile2D(dailyConfirmedCasesHigh, 0.75))],
        "k--", label=Dates.format(ALERT__CHANGE__DATE, "U d Y"), lw=2, alpha = 1)
    end


    if two21
        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
        122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
        184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]

        observedIDetect = diff(vcat([0],observedIDetect))
        tDetect = collect(0:length(observedIDetect)-1)

        plt.plot(tDetect, observedIDetect, color="tab:gray", linestyle="-", label="August 2020 Daily Detected Cases", lw=2.5, figure=fig)
    end

    if useDates
        plt.xticks(collect(dailyTimes[1]:10:dailyTimes[end]),
            [Dates.format(DAY__ZERO__DATE + Dates.Day(i*10), "u d") for i in 0:(length(collect(dailyTimes[1]:10:dailyTimes[end]))-1)])
        plt.xlabel("Date")

    else
        plt.xlabel("Days since detection")
    end

    if two21 && false
        plt.xticks([-3,0,3,6,9,12,15],vcat(["Aug $(14+3*i)" for i in 0:5], ["Sep 01"]))
        plt.xlabel("Date")
    else
        plt.xlabel("Days since Detection")
    end
    plt.ylabel("Daily Detected Cases")
    # plt.suptitle("Branching Process Simulation")
    plt.title(title)
    plt.legend(loc = "upper right")

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

function plotAndStatsOutbreak(confirmedCases, cumulativeCases, times, observedIDetect,
    tDetect, t_current, title, outputFileName, Display=true, save=false, alphaMultiplier=1.0,
    conditioned=false, useDates=false, two21=true, confirmedCases_ensemble=[], cumulativeCases_ensemble=[])
    #=
    Plot multiple realisations of confirmedCases and x2 as well as their means.
    =#

    Seaborn.set()
    set_style("ticks")
    set_color_codes("pastel")

    # Initialise plots - need figure size to make them square and nice
    f,ax = Seaborn.subplots(1,2, figsize=(12,6), dpi=300)

    timesVector = []
    for i in 1:length(confirmedCases[1,:])
        timesVector = vcat(timesVector, times)
    end

    type = ""
    # if conditioned
    #     type=" Curvewise"
    # end

    for i in 1:length(confirmedCases[1,:])
        if i == 1
            ax[1].plot(times, confirmedCases[:,i], "b-", label="BPM Realisations", lw=2, alpha = 0.2)
        else
            ax[1].plot(times, confirmedCases[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
        end
    end

    for i in 1:length(cumulativeCases[1,:])
        if i == 1
            ax[2].plot(times, cumulativeCases[:,i], "b-", label="BPM Realisations", lw=2, alpha = 0.2)
        else
            ax[2].plot(times, cumulativeCases[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
        end
    end


    # If using conditioned model ensemble, plot 50% (IQR) quantiles as well ----
    if conditioned

        if !isempty(confirmedCases_ensemble)
            for i in 1:min(700,length(confirmedCases_ensemble[1,:]))
                if i == 1
                    ax[1].plot(times, confirmedCases_ensemble[:,i], "k-", label="Unconditioned Realisations", lw=2, alpha=0.05)
                else
                    ax[1].plot(times, confirmedCases_ensemble[:,i], "k-", lw=2, alpha=0.09*alphaMultiplier)
                end
            end
        end

        if !isempty(cumulativeCases_ensemble)
            for i in 1:min(700,length(cumulativeCases_ensemble[1,:]))
                if i == 1
                    ax[2].plot(times, cumulativeCases_ensemble[:,i], "k-", label="Unconditioned Realisations", lw=2, alpha=0.05)
                else
                    ax[2].plot(times, cumulativeCases_ensemble[:,i], "k-", lw=2, alpha=0.09*alphaMultiplier)
                end
            end
        end

        # Confirmed cases since first detect ###################################
        quantiles95_1 = quantile2D(confirmedCases, 0.025)
        quantiles95_2 = quantile2D(confirmedCases, 0.975)
        quantiles50_1 = quantile2D(confirmedCases, 0.25)
        quantiles50_2 = quantile2D(confirmedCases, 0.75)

        # plotting 95% bands ---------------------------------------------------
        ax[1].plot(times, quantiles95_1, "k--", label="95%$type Quantile Bands ", lw=2, alpha = 0.5)
        ax[1].plot(times, quantiles95_2, "k--", lw=2, alpha = 0.5)

        ax[1].fill_between(times, quantiles95_1, quantiles50_1, alpha=0.3, color = "r")
        ax[1].fill_between(times, quantiles95_2, quantiles50_2, alpha=0.3, color = "r")

        # plotting 50% bands ---------------------------------------------------
        ax[1].plot(times, quantiles50_1, "r--", label="50%$type Quantile Bands ", lw=2, alpha = 0.8)
        ax[1].plot(times, quantiles50_2, "r--", lw=2, alpha = 0.8)
        ax[1].fill_between(times, quantiles50_1, quantiles50_2, alpha=0.3, color = "b")
        ########################################################################

        # Cumulative cases since first detect ##################################
        quantiles95_1 = quantile2D(cumulativeCases, 0.025)
        quantiles95_2 = quantile2D(cumulativeCases, 0.975)
        quantiles50_1 = quantile2D(cumulativeCases, 0.25)
        quantiles50_2 = quantile2D(cumulativeCases, 0.75)

        # plotting 95% bands ---------------------------------------------------
        ax[2].plot(times, quantiles95_1, "k--", label="95%$type Quantile Bands ", lw=2, alpha = 0.5)
        ax[2].plot(times, quantiles95_2, "k--", lw=2, alpha = 0.5)

        ax[2].fill_between(times, quantiles95_1, quantiles50_1, alpha=0.3, color = "r")
        ax[2].fill_between(times, quantiles95_2, quantiles50_2, alpha=0.3, color = "r")

        # plotting 50% bands ---------------------------------------------------
        ax[2].plot(times, quantiles50_1, "r--", label="50%$type Quantile Bands ", lw=2, alpha = 0.7)
        ax[2].plot(times, quantiles50_2, "r--", lw=2, alpha = 0.7)
        ax[2].fill_between(times, quantiles50_1, quantiles50_2, alpha=0.3, color = "b")
        ########################################################################

    else

        quantiles95_1 = quantile2D(confirmedCases, 0.025)
        quantiles95_2 = quantile2D(confirmedCases, 0.975)
        ax[1].fill_between(times, quantiles95_1, quantiles95_2, alpha=0.3, color = "r")

        quantiles95_1 = quantile2D(cumulativeCases, 0.025)
        quantiles95_2 = quantile2D(cumulativeCases, 0.975)
        ax[2].fill_between(times, quantiles95_1, quantiles95_2, alpha=0.3, color = "r")

    end


    ax[1].plot(times, median(confirmedCases, dims=2), "r-", label="Median Realisation", lw=3, alpha = 1)

    # 2020 Detected Cases
    if two21

        observedIDetect2020, tDetect2020 = observedCasesAugust2020(false)

        ax[1].plot(tDetect2020, observedIDetect2020, color="b", linestyle="-.", label="August Outbreak 2020", lw=2, alpha = 1)
    end

    ax[1].plot(tDetect, observedIDetect, "k-", label="August Outbreak", lw=2, alpha = 1)
    ax[2].plot(times, median(cumulativeCases, dims=2), "r-", label="Median Realisation", lw=3, alpha = 1)


    # plot the current date as a vertical line on the plot. t_current is a number
    # which
    if t_current > 0
        ax[1].plot([t_current, t_current], [minimum(observedIDetect), maximum(cumulativeCases)],
        "k--", label=Dates.format(DAY__ZERO__DATE+Dates.Day(t_current), "U d Y"), lw=2, alpha = 1)
        ax[2].plot([t_current, t_current], [minimum(observedIDetect), maximum(cumulativeCases)],
        "k--", label=Dates.format(DAY__ZERO__DATE+Dates.Day(t_current), "U d Y"), lw=2, alpha = 1)
    end

    # plot date of AL change from AL4 to AL3
    if CHANGING__ALERT__LEVELS
        t_change = daysFromZeroToALChange()

        ax[1].plot([t_change, t_change], [minimum(observedIDetect), maximum(cumulativeCases)],
        "k--", label=Dates.format(ALERT__CHANGE__DATE, "U d Y"), lw=2, alpha = 1)
        ax[2].plot([t_change, t_change], [minimum(observedIDetect), maximum(cumulativeCases)],
        "k--", label=Dates.format(ALERT__CHANGE__DATE, "U d Y"), lw=2, alpha = 1)
    end

    # ax[1].plot(times, median(confirmedCases, dims=2), "b-", label="Median Realisation", lw=4, alpha = 1)


    # Seaborn.lineplot(x = timesVector, y = [cumulativeCases...], hue = labelcumulativeCases, palette = "flare", ci="sd", ax=ax[2], estimator=median)




    # ax[1].axvline(10, "k--", label="21/8/2021")
    # ax[2].vline()

    # BP_df = DataFrame()
    # BP_df.timesVector = timesVector
    # BP_df.infections = [confirmedCases...]
    # BP_df.label = labelconfirmedCases
    #
    # filtered_BP = filter(row -> row.timesVector in timesOfInterest, BP_df, view=false)
    #
    # # Seaborn.violinplot(x=filtered_BP.timesVector, y=filtered_BP.infections, hue=filtered_BP.timesVector,
    #     # bw=0.4, cut=0, scale="count",palette = "Set2",ax=ax[2])
    #
    #
    # for i in timesOfInterest
    #     # meanNumber = round(mean(filter(row -> row.timesVector == i, filtered_BP).infections))
    #     # println("Mean number of infections is: $meanNumber")
    #     println("Summary Statistics at Day #$i")
    #     describe(filter(row -> row.timesVector == i, filtered_BP).infections)
    # end
    #
    # t_span = convert.(Int64, ceil.([times[1], times[end]]))
    #
    # c_mean = Array{Float64}(undef, t_span[end]-t_span[1]+1)
    # c_median = Array{Float64}(undef, t_span[end]-t_span[1]+1)
    #
    # timeDays = collect(t_span[1]:t_span[end])
    #
    # for i in 1:length(timeDays)
    #     c_mean[i] = mean(filter(row -> row.timesVector == float(timeDays[i]), BP_df).infections)
    #     c_median[i] = median(filter(row -> row.timesVector == float(timeDays[i]), BP_df).infections)
    # end
    #
    # growth_rates_mean = round.(diff(log.(diff(c_mean))), digits=2)
    # growth_rates_median = round.(diff(log.(diff(c_median))), digits=2)
    #
    # R_eff_mean = round.(1 .+ 5 .* diff(log.(diff(c_mean))), digits=2)
    # R_eff_median = round.(1 .+ 5 .* diff(log.(diff(c_median))), digits=2)
    #
    # println("Growth rate mean is $growth_rates_mean")
    # println("Growth rate median is $growth_rates_median")
    #
    #
    # println("Estimated mean R_eff is $R_eff_mean")
    # println("Estimated median R_eff is $R_eff_median")

    # maxy = maximum(confirmedCases)
    # maxy=1000

    # ax[1].set_ylim([0, maxy*0.6])

    if useDates
        ax[1].set_xticks(collect(times[1]:10:times[end]))
        ax[1].set_xticklabels([Dates.format(DAY__ZERO__DATE + Dates.Day(i*10), "u d") for i in 0:(length(collect(times[1]:10:times[end]))-1)])
        ax[1].set_xlabel("Date")

        ax[2].set_xticks(collect(times[1]:10:times[end]))
        ax[2].set_xticklabels([Dates.format(DAY__ZERO__DATE + Dates.Day(i*10), "u d") for i in 0:(length(collect(times[1]:10:times[end]))-1)])
        ax[2].set_xlabel("Date")

    else
        ax[1].set_xlabel("Days since detection")
        ax[2].set_xlabel("Days since detection")
    end

    if two21
        ax[1].set(yscale="log")
        ax[2].set(yscale="log")
        ax[1].set_yticks([1.0,10.0,100.0,1000.0,10000.0])
        ax[1].set_yticklabels([1,10,100,1000,10000])
        ax[2].set_yticks([1.0,10.0,100.0,1000.0, 10000.0])
        ax[2].set_yticklabels([1,10,100, 1000,10000])
    else
        ax[1].set(yscale="log")
        ax[2].set(yscale="log")
        ax[1].set_yticks([1.0,10.0,100.0,1000.0,10000.0])
        ax[1].set_yticklabels([1,10,100,1000,10000])
        ax[2].set_yticks([1.0,10.0,100.0,1000.0,10000.0])
        ax[2].set_yticklabels([1,10,100, 1000,10000])
        ax[1].set_ylim([0,1000])
        ax[2].set_ylim([0,1000])
    end

    ax[1].legend(loc = "lower right")
    ax[1].set_ylabel("Cumulative Detected Cases")
    ax[1].set_title("Detected Cases")


    # ax[2].set_ylim([0, maxy*0.6])
    ax[2].legend(loc = "lower right")
    ax[2].set_ylabel("Cumulative Total Cases")
    ax[2].set_title("Total Cases")

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

function plotReff(models_df, conditionIndexes, io, title, outputFileName, Display=true, save=false, useEstimate=false)

    Seaborn.set()
    Seaborn.set_style("ticks")
    # fig = plt.figure(dpi=300)
    fig,ax = Seaborn.subplots(1,2, figsize=(12,6), dpi=300)

    randIndexes = sample(collect(1:nrow(models_df)), length(conditionIndexes), replace=false)

    Reff1 = []; Reff2 = [];

    # Reff1 = vcat(models_df.R_eff_before_al[randIndexes], models_df.R_eff_after_al[randIndexes])
    # Reff2 = vcat(models_df.R_eff_before_al[conditionIndexes], models_df.R_eff_after_al[conditionIndexes])
    # Reff = vcat(Reff1, Reff2)
    if useEstimate
        Reff1 = vcat(models_df.R_eff_before_al[randIndexes], models_df.R_eff_before_al[conditionIndexes])
        Reff2 = vcat(models_df.R_eff_after_al_estimate[randIndexes], models_df.R_eff_after_al_estimate[conditionIndexes])
    else
        Reff1 = vcat(models_df.R_eff_before_al[randIndexes], models_df.R_eff_before_al[conditionIndexes])
        Reff2 = vcat(models_df.R_eff_after_al[randIndexes], models_df.R_eff_after_al[conditionIndexes])
    end
    # x_vector = vcat(["Before" for _ in 1:length(conditionIndexes)], ["After" for _ in 1:length(conditionIndexes)])
    # x_vector2 = vcat(["Before" for _ in 1:length(conditionIndexes)], ["After" for _ in 1:length(conditionIndexes)])
    # x_vector = vcat(x_vector,x_vector2)

    x_vector1 = vcat(["Before" for _ in 1:length(Reff1)])
    x_vector2 = vcat(["After" for _ in 1:length(Reff1)])


    # hue_vector = ["Unconditioned" for _ in 1:length(Reff1)]
    # hue_vector = vcat(hue_vector, ["Conditioned" for _ in 1:length(conditionIndexes)*2])

    hue_vector = vcat(["Unconditioned" for _ in 1:length(randIndexes)], ["Conditioned" for _ in 1:length(conditionIndexes)])

    # println(length(Reff))
    # println(length(x_vector))
    # println(length(hue_vector))
    #
    # @assert length(x_vector) == length(hue_vector)
    # @assert length(x_vector) == length(Reff)

    Seaborn.violinplot(x=x_vector1, y=Reff1, hue=hue_vector, bw=0.2, cut=0.0, scale="count", palette = "Set2", aspect=.7, ax=ax[1], gridsize=40)
    Seaborn.violinplot(x=x_vector2, y=Reff2, hue=hue_vector, bw=0.2, cut=0.0, scale="count", palette = "Set2", aspect=.7, ax=ax[2], gridsize=40)


    # Seaborn.swarmplot(x=x_vector, y=Reff, hue=hue_vector)

    # ax[1].set_xlabel("Before Alert Level")
    # ax[2].set_xlabel("After Alert Level")
    ax[1].set_ylabel("Effective Reproduction Number")
    ax[2].set_ylabel("Effective Reproduction Number")

    # plt.xlabel("Before or After Alert Level")
    # plt.ylabel("Effective Reproduction Number")
    plt.suptitle(title)
    # # plt.title("For alpha = $alpha and beta $beta")
    ax[2].legend(loc = "upper right")
    ax[1].legend().set_visible(false)

    plt.tight_layout(h_pad=0.01)

    # fig.savefig(outputFileName)

    # plt.xticks([0,1],["Before, After"])

    if Display
        # required to display graph on plots.
        display(fig)
    end
    if save
        # Save graph as pngW
        fig.savefig(outputFileName)

    end
    close()


    plotReffHist(models_df.R_eff_after_al[conditionIndexes], "Conditioned value of Reff post Alert Level", outputFileName*"_hist_py", Display, save)

    # set up figure # histogram of Reff in AL4
    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(figsize=(10,6),dpi=300)

    Seaborn.hist(x=models_df.R_eff_after_al[conditionIndexes], bins=16, weights=ones(length(conditionIndexes))/length(conditionIndexes))

    # else
    # end
    plt.xlabel("Reff")
    plt.ylabel("Probability")
    # plt.suptitle("Branching Process Simulation")
    plt.title("Conditioned value of Reff post Alert Level")
    # plt.legend(loc = "upper right")

    if Display
        # required to display graph on plots.
        display(fig)
    end
    if save
        # Save graph as pngW
        fig.savefig(outputFileName*"_hist")

    end
    close()

    # println(io, "The median value of \$R_\\text{eff}\$ is $(median(models_df.R_eff_after_al[conditionIndexes])) with IQR between [$(quantile(models_df.R_eff_after_al[conditionIndexes], 0.25)), $(quantile(models_df.R_eff_after_al[conditionIndexes], 0.75))]" )

    # println(io, "\nWe estimate that the probability of \$R_\\text{eff}\$ being less than 1 is: $(round(sum(models_df.R_eff_after_al[conditionIndexes] .< 1.0)/length(conditionIndexes),sigdigits=3))")

    return io
end

# function plotReffAL3(x_vector, hue_vector, io, title, outputFileName, Display=true, save=false)
#
#     # set up figure # histogram of Reff in AL4
#     Seaborn.set()
#     set_style("ticks")
#     Seaborn.set_color_codes("pastel")
#     fig = plt.figure(figsize=(10,6),dpi=300)
#
#     # Seaborn.hist(x=models_df.R_eff_after_al[conditionIndexes], bins=16, weights=ones(length(conditionIndexes))/length(conditionIndexes))
#
#     Seaborn.distplot(x=x_vector,hist=false,kde=true)
#
#     # else
#     # end
#     plt.xlabel("Reff")
#     plt.ylabel("Probability")
#
#     plt.title("Conditioned value of Reff post Alert Level")
#     plt.legend(loc = "upper right")
#
#     if Display
#         # required to display graph on plots.
#         display(fig)
#     end
#     if save
#         # Save graph as pngW
#         fig.savefig(outputFileName*"_hist")
#
#     end
#     close()
#
#     # println(io, "\nWe estimate that the probability of \$R_\\text{eff}\$ being less than 1 is: $(round(sum(models_df.R_eff_after_al[conditionIndexes] .< 1.0)/length(conditionIndexes),sigdigits=3))")
#
#     return io
# end

function simulateHospDaily(dailyTimes, dailyDetectedCases, propHosp, propofHosp_critical, propofCritical_recover)

    # dailyDetectedCases is the 2d array of all realisations
    dailyDetectedCases = convert.(Int64,round.(dailyDetectedCases))

    numSimsPerRealisation = 10

    dailyHosp = Array{Int64}(undef, length(dailyTimes), length(dailyDetectedCases[1,:])*numSimsPerRealisation) .* 0
    dailyCrit = Array{Int64}(undef, length(dailyTimes), length(dailyDetectedCases[1,:])*numSimsPerRealisation) .* 0

    hospIndex = 1
    @showprogress PROGRESS__METER__DT for i in 1:numSimsPerRealisation
        for col in 1:length(dailyDetectedCases[1,:])
            totalDetected = sum(dailyDetectedCases[:,col])

            detected_df = DataFrame()

            hospEntryTime = Array{Float64}(undef, totalDetected) .* 0
            hospExitTime = Array{Float64}(undef, totalDetected) .* 0
            critEntryTime = Array{Float64}(undef, totalDetected) .* 0
            hospCase = Array{Bool}(undef, totalDetected) .* false
            critCase = Array{Bool}(undef, totalDetected) .* false

            detected_df.hospEntryTime = hospEntryTime
            detected_df.hospExitTime = hospExitTime
            detected_df.critEntryTime = critEntryTime
            detected_df.hospCase = hospCase
            detected_df.critCase = critCase
            detected_df.critCaseR = critCase .* false # i.e. is true if recover, false if death

            # simulate whether case is hospitalised
            detected_df.hospCase .= rand(Bernoulli(propHosp), totalDetected)

            # simulate whether hosp case is critical
            hospCases_df = filter(row -> row.hospCase, detected_df, view=true)
            hospCases_df.critCase .= rand(Bernoulli(propofHosp_critical), nrow(hospCases_df))

            critCases_df = filter(row -> row.critCase, detected_df, view=true)
            critCases_df.critCaseR .= rand(Bernoulli(propofCritical_recover), nrow(critCases_df))

            # entry times
            currentIndex = 1
            currentIndex_t = 1
            for j in dailyDetectedCases[:, col]

                a = dailyTimes[currentIndex_t]*1
                detected_df[currentIndex:currentIndex+j-1, :hospEntryTime] .= dailyTimes[currentIndex_t]*1

                currentIndex += j
                currentIndex_t += 1
            end

            # critEntryDist = Normal(2, 0.5)
            # noncritExitDist = Normal(6, 0.5)
            # critExitDist_D = Normal(9, 0.5)
            # critExitDist_R = Normal(16, 0.5)

            critEntryDist = Exponential(2)
            noncritExitDist = Exponential(6)
            critExitDist_D = Exponential(9)
            critExitDist_R = Exponential(16)

            detected_df.critEntryTime .= rand(critEntryDist, totalDetected)

            # exit times
            detected_df.hospExitTime = detected_df.hospCase .* (detected_df.hospEntryTime .+
                .!detected_df.critCase .* rand(noncritExitDist, totalDetected) .+
                detected_df.critCase .* (detected_df.critEntryTime .+
                detected_df.critCaseR .* rand(critExitDist_R, totalDetected) .+
                .!detected_df.critCaseR .* rand(critExitDist_D, totalDetected)
                ))

            for k in 1:length(dailyTimes)

                currentHosp_df = filter(row -> row.hospCase && row.hospEntryTime <= dailyTimes[k] &&
                    row.hospExitTime >= dailyTimes[k], detected_df, view=true)

                currentCrit_df = filter(row -> row.critCase && (row.critEntryTime .+row.hospEntryTime) < dailyTimes[k] &&
                    row.hospExitTime >= dailyTimes[k], currentHosp_df, view=true)

                dailyHosp[k, hospIndex] = nrow(currentHosp_df)
                dailyCrit[k, hospIndex] = nrow(currentCrit_df)

            end

            hospIndex +=1
        end
    end

    return dailyHosp, dailyCrit
end

function plotHospEstimate(dailyTimes, cumulativeConfirmedCases, proportion, title, outputFileName, Display=true, save=false)

    # set up figure
    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(figsize=(8,6),dpi=300)

    cumulativeConfirmedCases = cumulativeConfirmedCases .* proportion


    quantiles95_1 = quantile2D(cumulativeConfirmedCases, 0.025)
    quantiles95_2 = quantile2D(cumulativeConfirmedCases, 0.975)
    quantiles50_1 = quantile2D(cumulativeConfirmedCases, 0.25)
    quantiles50_2 = quantile2D(cumulativeConfirmedCases, 0.75)

    # plotting 95% bands ---------------------------------------------------
    plt.plot(dailyTimes, quantiles95_1, "k--", label="95% Curvewise Quantile Bands ", lw=2, alpha = 0.5)
    plt.plot(dailyTimes, quantiles95_2, "k--", lw=2, alpha = 0.5)

    plt.fill_between(dailyTimes, quantiles95_1, quantiles50_1, alpha=0.5, color = "r")
    plt.fill_between(dailyTimes, quantiles95_2, quantiles50_2, alpha=0.5, color = "r")

    # plotting 50% bands ---------------------------------------------------
    plt.plot(dailyTimes, quantiles50_1, "r--", label="50% Curvewise Quantile Bands", lw=2, alpha = 0.8)
    plt.plot(dailyTimes, quantiles50_2, "r--", lw=2, alpha = 0.8)
    plt.fill_between(dailyTimes, quantiles50_1, quantiles50_2, alpha=0.5, color = "b")

    plt.plot(dailyTimes, median(cumulativeConfirmedCases,dims=2), "r-", label="Median cumulative hospitalisations", lw=2.5, figure=fig)

    # if useDates
    plt.xticks(collect(dailyTimes[1]:10:dailyTimes[end]),
        [Dates.format(DAY__ZERO__DATE + Dates.Day(i*10), "u d") for i in 0:(length(collect(dailyTimes[1]:10:dailyTimes[end]))-1)])
    plt.xlabel("Date")

    # else
    #     plt.xlabel("Days since detection")
    # end
    plt.ylabel("Cumulative Estimated Hospitalisations")
    # plt.suptitle("Branching Process Simulation")
    plt.title(title)
    plt.legend(loc = "upper right")

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

function plotHospEstimateDaily(dailyTimes, dailyConfirmedCases, title, outputFileName, Display=true, save=false)

    # set up figure
    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(figsize=(8,6),dpi=300)

    quantiles95_1 = quantile2D(dailyConfirmedCases, 0.025)
    quantiles95_2 = quantile2D(dailyConfirmedCases, 0.975)
    quantiles50_1 = quantile2D(dailyConfirmedCases, 0.25)
    quantiles50_2 = quantile2D(dailyConfirmedCases, 0.75)

    # plotting 95% bands ---------------------------------------------------
    plt.plot(dailyTimes, quantiles95_1, "k--", label="95% Quantile Bands ", lw=2, alpha = 0.5)
    plt.plot(dailyTimes, quantiles95_2, "k--", lw=2, alpha = 0.5)

    plt.fill_between(dailyTimes, quantiles95_1, quantiles50_1, alpha=0.5, color = "r")
    plt.fill_between(dailyTimes, quantiles95_2, quantiles50_2, alpha=0.5, color = "r")

    # plotting 50% bands ---------------------------------------------------
    plt.plot(dailyTimes, quantiles50_1, "r--", label="50% Quantile Bands", lw=2, alpha = 0.8)
    plt.plot(dailyTimes, quantiles50_2, "r--", lw=2, alpha = 0.8)
    plt.fill_between(dailyTimes, quantiles50_1, quantiles50_2, alpha=0.5, color = "b")

    plt.plot(dailyTimes, median(dailyConfirmedCases,dims=2), "r-", label="Median daily in hospital", lw=2.5, figure=fig)

    # if useDates
    plt.xticks(collect(dailyTimes[1]:10:dailyTimes[end]),
        [Dates.format(DAY__ZERO__DATE + Dates.Day(i*10), "u d") for i in 0:(length(collect(dailyTimes[1]:10:dailyTimes[end]))-1)])
    plt.xlabel("Date")

    # else
    #     plt.xlabel("Days since detection")
    # end
    plt.ylabel("Number of people")
    # plt.suptitle("Branching Process Simulation")
    plt.title(title)
    plt.legend(loc = "upper right")

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

function probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange,
    title, outputFileName, Display=true, save=false)

    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(dpi=300)


    Seaborn.heatmap(probabilities, cmap="rocket")


    # plt.xticks([-3,0,3,6,9,12,15],vcat(["Aug $(14+3*i)" for i in 0:5], ["Sep 01"]))
    # no more than 8 labels



    plt.yticks(collect(1:length(numCasesRange)) , collect(numCasesRange))
    plt.ylabel("Less than N cases per day")

    plt.xlabel("Days since detection")
    plt.xticks(collect(1:length(daysSinceRange)) , collect(daysSinceRange))
    # plt.suptitle("Branching Process Simulation")
    plt.title(title)
    # plt.legend()

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

function probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange,
    title, outputFileName, Display=true, save=false, useDates=false)

    caseData, times = observedCases()
    caseData = diff(vcat([0],caseData))
    # caseData = [5,4,16,20,27,36,37,60,69,79,76,79,54,65,60]

    index = findfirst(daysSinceRange[1] .== times)+2

    caseData = caseData[index:end]
    times = times[index:end]

    xValues = []
    xTitle = ""
    if useDates
        # dayZeroDate = Date("17-08-2021", dateformat"d-m-y")
        xValues = [Dates.format(DAY__ZERO__DATE + Dates.Day(i), "u d") for i in collect(daysSinceRange)]
        xTitle = "Date"

        times = [Dates.format(DAY__ZERO__DATE + Dates.Day(i), "u d") for i in times]
    else
        xValues = collect(daysSinceRange)
        xTitle = "Days since detection"
    end




    fig = PlotlyJS.plot([PlotlyJS.contour(
        x=xValues, # horizontal axis
        y=collect(numCasesRange), # vertical axis
        z=probabilities,
        # heatmap gradient coloring is applied between each contour level
        contours=attr(
            coloring ="heatmap",
            showlabels = true, # show labels on contours
            labelfont = attr( # label font properties
                size = 12,
                color = "white",
            )
        ), colorbar=attr(
        nticks=10, ticks="outside",
        ticklen=5, tickwidth=2,
        showticklabels=true,
        tickangle=0, tickfont_size=15,title="Probability", # title here
        titleside="right",
        titlefont=attr(
            size=18,
            family="Arial, sans-serif"
        )
        )),

        PlotlyJS.scatter(
        x=times, y=caseData, mode="markers+lines",
        name="Daily Case Data", line_color="white"
        )

    ], Layout(width=5*150, height=4*150, font=attr(
        size=16,
        family="Arial, sans-serif"
    ), title=title,
    yaxis=attr(title_font=attr(size=16), ticks="outside", tickwidth=2, ticklen=5, col=1,showline=true, linewidth=2, linecolor="black", mirror=true),
    xaxis=attr(title_font=attr(size=16), nticks=7, ticks="outside", tickwidth=2, ticklen=5, col=1, showline=true, linewidth=2, linecolor="black", mirror=true),
    xaxis_title=xTitle, yaxis_title="Less than N cases per day", scene_xaxis_range=[xValues[1], xValues[end]],
    showlegend=true, legend=attr(
        xanchor= "left",
        x=0,
        y=0,
        traceorder="reversed",
        title_font_family="Arial, sans-serif",
        font=attr(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    )

    ))

    # fig = PlotlyJS.plot([
    #     PlotlyJS.contour(
    #         x=xValues, # horizontal axis
    #         y=collect(numCasesRange), # vertical axis
    #         z=probabilities,
    #         ncontours=30,
    #         showscale=false
    #     ),
    #     PlotlyJS.scatter(
    #         x=times, y=caseData, mode="markers+lines",
    #         name="steepest", line_color="black"
    #     )
    #     ])


    # if Display
    #     # required to display graph on plots.
    #     display(fig)
    # end
    if save
        # Save graph as pngW
        PlotlyJS.savefig(fig, outputFileName*".png", width=5*150, height=4*150,scale=2)

    end
    close()
end

function overleafTableFormat(datesString)
    newDatesString = datesString[1]
    for date in datesString[2:end]
        newDatesString = newDatesString * " & " * date
    end
    return newDatesString * "\\\\"
end

function caseNumbersBeginToDrop(dailyConfirmedCases, io, t_current, numCasesRange=0:1, daysSinceRange=0:1)

    if t_current != -1
        # println(io, "\nDaily cases are projected to drop:")
        #
        # maxVal, maxIndex = findmax(median(dailyConfirmedCases, dims=2))
        #
        medianDCC = median(dailyConfirmedCases, dims=2)
        lowerQCC = quantile2D(dailyConfirmedCases, 0.25)
        upperQCC = quantile2D(dailyConfirmedCases, 0.75)
        #
        # println(io, "Cases numbers on average begin to drop $(maxIndex[1]) days after detection on $(Dates.format(DAY__ZERO__DATE + Dates.Day(maxIndex[1]), "u d"))")
        # println(io, "Median max case number is $(round(maxVal))")
        #
        #
        # maxVal, maxIndex = findmax(quantile2D(dailyConfirmedCases, 0.75))
        #
        # println(io, "Cases numbers 75% quartile begin to drop $(maxIndex[1]) days after detection on $(Dates.format(DAY__ZERO__DATE + Dates.Day(maxIndex[1]), "u d"))")
        # println(io, "75% Quartile max case number is $(round(maxVal))")
        #
        # maxVal, maxIndex = findmax(quantile2D(dailyConfirmedCases, 0.25))
        #
        # println(io, "Cases numbers 25% quartile begin to drop $(maxIndex[1]) days after detection on $(Dates.format(DAY__ZERO__DATE + Dates.Day(maxIndex[1]), "u d"))")
        # println(io, "25% Quartile max case number is $(round(maxVal))")
        #
        # println(io, "\nProjected cases for the next 5 days are:")
        # for i in t_current+1:t_current+6
        #     println(io, "$(Dates.format(DAY__ZERO__DATE+Dates.Day(i), "U d Y")): $(round(medianDCC[i])), IQR = [$(round(lowerQCC[i])),$(round(upperQCC[i]))]")
        # end

        # println(io, "\nProjected cases for the next 5 days are (overleaf table format):")
        println(io, "\n\\begin{table}[h] \n \\centering \n \\begin{tabular}{c|c|c|c}")

        t_range = t_current .+ [1, 3, 7] .+ 1 # need plus 1 because indexing from 1 not zero

        datesString = [Dates.format(DAY__ZERO__DATE+Dates.Day(i), "U d Y") for i in t_range]

        newDatesString = overleafTableFormat(datesString)
        # for date in datesString[2:end]
        #     newDatesString = newDatesString * " & " * date
        # end

        println(io, "Data Quartiles"*" & "*newDatesString)
        println(io, "\\hline")
        println(io, "25\\% Quartile"*" & "*overleafTableFormat(string.(round.(lowerQCC[t_range]))))
        println(io, "Median"*" & "*overleafTableFormat(string.(round.(medianDCC[t_range]))))
        println(io, "75\\% Quartile"*" & "*overleafTableFormat(string.(round.(upperQCC[t_range]))))
        println(io, "\\end{tabular}\n\\caption{Projected cases in the next 1, 3 and 7 days}\n\\label{tab:BP_predicted_cases}\n\\end{table}")
    end

    # numCasesRange = 10:60
    # daysSinceRange = 45:70
    probabilities = zeros(length(numCasesRange), length(daysSinceRange))

    for i in 1:length(numCasesRange)
        probabilities[i,:] = probOfLessThanXGivenYDays(dailyConfirmedCases, numCasesRange[i], daysSinceRange)
    end


    return probabilities, io
end

function nothingConvert(input)
    if isnothing(input)
        return 1000
    end
    return input
end

function caseNumbersHitValue(dailyConfirmedCases, io, value)


    # println(io, "\nCases numbers are projected to hit $value per day by:")

    maxVal, maxIndex = findmax(median(dailyConfirmedCases, dims=2))

    medianDCC = median(dailyConfirmedCases, dims=2)
    lowerQCC = quantile2D(dailyConfirmedCases, 0.25)
    upperQCC = quantile2D(dailyConfirmedCases, 0.75)
    lower95CC = quantile2D(dailyConfirmedCases, 0.025)
    upper95CC = quantile2D(dailyConfirmedCases, 0.975)

    indexMed = nothingConvert(findfirst(medianDCC[11:end] .<= value)) + 10
    indexLQCC = nothingConvert(findfirst(lowerQCC[11:end] .<= value)) + 10
    indexUQCC = nothingConvert(findfirst(upperQCC[11:end] .<= value)) + 10
    indexL95CC = nothingConvert(findfirst(lower95CC[11:end] .<= value)) + 10
    indexU95CC = nothingConvert(findfirst(upper95CC[11:end] .<= value)) + 10

    # println(io, "On average: $(Dates.format(DAY__ZERO__DATE + Dates.Day(indexMed), "U d Y"))")
    # println(io, "IQR = [$(Dates.format(DAY__ZERO__DATE + Dates.Day(indexLQCC),"U d Y")), $(Dates.format(DAY__ZERO__DATE + Dates.Day(indexUQCC),"U d Y"))]")
    # println(io, "95% Quartile Range = [$(Dates.format(DAY__ZERO__DATE + Dates.Day(indexL95CC),"U d Y")), $(Dates.format(DAY__ZERO__DATE + Dates.Day(indexU95CC),"U d Y"))]")
    # println(io, "\nProjected cases for the next 5 days are (overleaf table format):")
    # println(io, "\\begin{table}[h] \n \\centering \n \\begin{tabular}{c|c|c|c|c|c}")
    #
    # datesString = [Dates.format(DAY__ZERO__DATE+Dates.Day(i), "U d Y") for i in t_current+1:t_current+6]
    #
    # newDatesString = overleafTableFormat(datesString)
    # # for date in datesString[2:end]
    # #     newDatesString = newDatesString * " & " * date
    # # end
    #
    # println(io, "Data Quartiles"*" & "*newDatesString)
    # println(io, "\\hline")
    # println(io, "25\\% Quartile"*" & "*overleafTableFormat(string.(round.(lowerQCC[t_current+1:t_current+6]))))
    # println(io, "Median"*" & "*overleafTableFormat(string.(round.(medianDCC[t_current+1:t_current+6]))))
    # println(io, "75\\% Quartile"*" & "*overleafTableFormat(string.(round.(upperQCC[t_current+1:t_current+6]))))
    # println(io, "\\end{tabular}\n\\caption{}\n\\label{tab:}\n\\end{table}")




    println(io, "\n\\begin{table}[h] \n \\centering \n \\begin{tabular}{c|c|c|c|c|c}")

    t_range = [indexL95CC, indexLQCC, indexMed, indexUQCC, indexU95CC]

    datesString = [Dates.format(DAY__ZERO__DATE+Dates.Day(i), "U d Y") for i in t_range]

    newDatesString = overleafTableFormat(datesString)
    # for date in datesString[2:end]
    #     newDatesString = newDatesString * " & " * date
    # end

    println(io, "Number of Cases & 2.5\\% Quartile & 25\\% Quartile & Median & 75\\% Quartile & 97.5\\% Quartile \\\\")
    println(io, "\\hline")
    println(io, "$value"*" & "*newDatesString)


    # println(io, "25\\% Quartile"*" & "*overleafTableFormat(string.(round.(lowerQCC[t_range]))))
    # println(io, "Median"*" & "*overleafTableFormat(string.(round.(medianDCC[t_range]))))
    # println(io, "75\\% Quartile"*" & "*overleafTableFormat(string.(round.(upperQCC[t_range]))))
    println(io, "\\end{tabular}\n\\caption{Projected date to reach $value cases}\n\\label{tab:BP_date_to_reach_cases}\n\\end{table}")

    return io
end

function caseChangeAfterAlertChange(dailyConfirmedCases, io, alertIncrease)


    # println(io, "\nDaily cases are projected to drop:")
    #
    # maxVal, maxIndex = findmax(median(dailyConfirmedCases, dims=2))
    #
    t_ind = daysFromZeroToALChange()

    # case numbers on 16 Sep


    medianDCC = median(dailyConfirmedCases, dims=2)
    lowerQCC = quantile2D(dailyConfirmedCases, 0.25)
    upperQCC = quantile2D(dailyConfirmedCases, 0.75)

    println(io, "\n\\begin{table}[h] \n \\centering \n \\begin{tabular}{c|c|c|c}")

    # t_range = t_current .+ [1, 3, 7] .+ 1 # need plus 1 because indexing from 1 not zero

    # datesString = [Dates.format(DAY__ZERO__DATE+Dates.Day(i), "U d Y") for i in t_range]

    # newDatesString = overleafTableFormat(datesString)

    values = [lowerQCC[t_ind], medianDCC[t_ind], upperQCC[t_ind]]


    println(io, "Date & 25\\% Quartile & Median & 75\\% Quartile \\\\")
    println(io, "\\hline")
    println(io, "$(Dates.format(ALERT__CHANGE__DATE, "U d Y"))"*" & "*overleafTableFormat(string.(round.(values))))
    # println(io, "Median"*" & "*overleafTableFormat(string.(round.(medianDCC[t_range]))))
    # println(io, "75\\% Quartile"*" & "*overleafTableFormat(string.(round.(upperQCC[t_range]))))
    println(io, "\\end{tabular}\n\\caption{Projected detected case numbers on $(Dates.format(ALERT__CHANGE__DATE, "U d Y"))}\n\\label{tab:BP_predicted_cases}\n\\end{table}")



    for increase in sort(unique(alertIncrease))

        filterIndx = findall(increase .== alertIncrease)

        medianDCC = median(dailyConfirmedCases[:, filterIndx], dims=2)
        lowerQCC = quantile2D(dailyConfirmedCases[:, filterIndx], 0.25)
        upperQCC = quantile2D(dailyConfirmedCases[:, filterIndx], 0.75)

        println(io, "\n\\begin{table}[h] \n \\centering \n \\begin{tabular}{c|c|c|c|c}")

        t_range = t_ind .+ [3, 7, 10, 14]

        datesString = [Dates.format(DAY__ZERO__DATE+Dates.Day(i), "U d Y") for i in t_range]

        newDatesString = overleafTableFormat(datesString)
        # for date in datesString[2:end]
        #     newDatesString = newDatesString * " & " * date
        # end

        println(io, "Data Quartiles"*" & "*newDatesString)
        println(io, "\\hline")
        # println(io, "25\\% Quartile"*" & "*overleafTableFormat(string.(round.(lowerQCC[t_range] .- lowerQCC[t_ind]))))
        # println(io, "Median"*" & "*overleafTableFormat(string.(round.(medianDCC[t_range] .- lowerQCC[t_ind]))))
        # println(io, "75\\% Quartile"*" & "*overleafTableFormat(string.(round.(upperQCC[t_range] .- lowerQCC[t_ind]))))
        println(io, "25\\% Quartile"*" & "*overleafTableFormat(string.(round.(lowerQCC[t_range] .- lowerQCC[t_ind]))))
        println(io, "Median"*" & "*overleafTableFormat(string.(round.(medianDCC[t_range] .- medianDCC[t_ind]))))
        println(io, "75\\% Quartile"*" & "*overleafTableFormat(string.(round.(upperQCC[t_range] .- upperQCC[t_ind]))))
        print(io, "\\end{tabular}\n\\caption{Relative change in daily cases from the value on $(Dates.format(ALERT__CHANGE__DATE, "U d Y")), ")
        print(io,"3, 7, 10 and 14 days after moving to Alert Level 3. For an increase in \$R_\\text{eff}\$ of $increase times. ")
        print(io, "If values in this table are positive, daily case numbers are increasing and vice versa.}\n\\label{tab:BP_predicted_cases_$increase}\n\\end{table}")

    end


    return io
end

function baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan, maxCases=20*10^3, p_test::Array{Float64,1}=[0.1,0.8], R_number=3, R_alert_scaling=0.2,
    t_onset_to_isol::Union{Array{Int64,1},Array{Float64,1}}=[2.2,1], initCases=1, num_detected_before_alert=1, alert_scaling_speed=5)
    StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)
    IDetect_tStep = StStep .* 0

    timesDaily = [i for i=tspan[1]-1:1:tspan[end]]
    tspanNew = (tspan[1],tspan[2]+1)
    dailyDetectCases, dailyICases, dailyRCases = initSIRArrays(tspanNew, 1, numSims)
    @assert length(dailyDetectCases[:,1]) == length(timesDaily)

    models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0], true) for i in 1:Threads.nthreads()]
    population_dfs = [initDataframe(models[1]) for i in 1:Threads.nthreads()];
    i = 1
    p = Progress(numSims,PROGRESS__METER__DT)

    meanOffspring = zeros(numSims)
    meanRNumber = zeros(numSims)

    establishedSims = BitArray(undef, numSims) .* false
    establishedSims .= true

    time = @elapsed Threads.@threads for i = 1:numSims

        models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^6, maxCases, [5*10^6-initCases,initCases,0], true);
        models[Threads.threadid()].stochasticRi = true
        models[Threads.threadid()].reproduction_number = R_number*1
        models[Threads.threadid()].p_test = p_test[1]*1
        models[Threads.threadid()].alert_pars.p_test = p_test[2]*1
        models[Threads.threadid()].t_onset_to_isol = t_onset_to_isol[1]*1
        models[Threads.threadid()].alert_pars.t_onset_to_isol = t_onset_to_isol[2]*1
        models[Threads.threadid()].alert_pars.R_scaling = R_alert_scaling*1
        models[Threads.threadid()].alert_pars.num_detected_before_alert = num_detected_before_alert*1
        models[Threads.threadid()].alert_pars.alert_level_scaling_speed = alert_scaling_speed

        # next React branching process with alert level
        population_dfs[Threads.threadid()] = initDataframe(models[Threads.threadid()]);
        t, state_totals_all, num_cases = nextReact_branch!(population_dfs[Threads.threadid()], models[Threads.threadid()])

        firstDetectIndex = findfirst(state_totals_all[:,4].==num_detected_before_alert)

        if !isnothing(firstDetectIndex)
            t_first_detect = t[firstDetectIndex]

            if detection_tspan[1] < t_first_detect && t_first_detect < detection_tspan[end]

                detectIndexes = findall(diff(state_totals_all[:,4]).==1) .+ 1
                undetectedIndexes = setdiff(collect(1:length(t)), detectIndexes)
                state_totals_all_new = state_totals_all[undetectedIndexes, 1:3]
                # tnew = t[undetectedIndexes] .- t_first_detect
                tnew = t .- t_first_detect

                IDetect = state_totals_all[detectIndexes, 4]
                tnew_detect = t[detectIndexes] .- t_first_detect


                StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, tnew, times)

                IDetect_tStep[:,i] = singleLinearSpline(state_totals_all[:, 4], tnew, times)
                dailyICases[:,i], dailyRCases[:,i], dailyDetectCases[:,i] = multipleLinearSplines(state_totals_all[:, 2:4], tnew, timesDaily)

            else
                establishedSims[i] = false
            end
        else
            establishedSims[i] = false
        end

        next!(p)
    end

    # dailyDetectCases = diff(hcat(convert.(Int64, zeros(length(timesDaily))), dailyDetectCases),dims=1)
    # dailyDetectCases = diff(vcat(transpose(zeros(numSims)), dailyDetectCases),dims=1)
    dailyDetectCases = diff(dailyDetectCases,dims=1)
    dailyTotalCases = dailyICases .+ dailyRCases
    dailyTotalCases = diff(dailyTotalCases, dims=1)



    IcumCases = ItStep .+ RtStep

    indexesToKeep = []
    sizehint!(indexesToKeep, numSims)

    for i in 2:length(establishedSims)
        if establishedSims[i]
            # xnew = hcat(xnew, x[:,i])
            push!(indexesToKeep, i)
        end
    end

    println("Kept $(length(indexesToKeep)) Sims or $(length(indexesToKeep)/numSims*100)% of Sims")

    # observedIDetect = [1, 10, 20, 30, 51]
    observedIDetect, tDetect = observedCases()

    return IcumCases[:,indexesToKeep], IDetect_tStep[:,indexesToKeep], observedIDetect, tDetect, dailyDetectCases[:,indexesToKeep], dailyTotalCases[:,indexesToKeep]
end

struct Ensemble
    p_test_ranges::Array
    p_test_sub_ranges::Array
    t_isol_ranges::Array
    alert_level_scaling_range::Array
    alert_level_speed_range::Array
    R_input_range::Array
    init_cases_range::Array{Int64,1}
    extra_alert_event::Dict{}
end

function ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan, ensemble::Ensemble, maxCases=20*10^3,
    initCases=1, num_detected_before_alert=1)

    StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)
    IDetect_tStep = StStep .* 0

    timesDaily = [i for i=tspan[1]-1:1:tspan[end]]
    tspanNew = (tspan[1],tspan[2]+1)
    dailyDetectCases, dailyICases, dailyRCases = initSIRArrays(tspanNew, 1, numSims)
    @assert length(dailyDetectCases[:,1]) == length(timesDaily)

    models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0], true) for i in 1:Threads.nthreads()]
    population_dfs = [initDataframe(models[1]) for i in 1:Threads.nthreads()];
    initCases = [rand(ensemble.init_cases_range)*1 for i in 1:Threads.nthreads()];

    p = Progress(numSims,PROGRESS__METER__DT)

    meanOffspring = zeros(numSims)
    meanRNumber = zeros(numSims)

    alertIncrease = zeros(numSims)

    establishedSims = BitArray(undef, numSims) .* false
    establishedSims .= true

    model_array = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0], true) for _ in 1:numSims]

    model_Reff_accepted = [false for _ in 1:Threads.nthreads()]

    i = 1
    time = @elapsed Threads.@threads for i = 1:numSims

        initCases[Threads.threadid()] = rand(ensemble.init_cases_range)*1

        model_Reff_accepted[Threads.threadid()] = false

        while !model_Reff_accepted[Threads.threadid()]

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^6, maxCases, [5*10^6-initCases[Threads.threadid()],initCases[Threads.threadid()],0], true);

            # shorten recovery time to 21 days from 30. Weibull dist effectively 1 by this stage.
            # should make detecting cases more realistic immediately after intervention
            models[Threads.threadid()].recovery_time = 21

            models[Threads.threadid()].stochasticRi = true
            models[Threads.threadid()].reproduction_number = rand(Uniform(ensemble.R_input_range...))
            models[Threads.threadid()].p_test = rand(Uniform(ensemble.p_test_ranges[1]...))
            models[Threads.threadid()].alert_pars.p_test = rand(Uniform(ensemble.p_test_ranges[2]...))

            if ensemble.p_test_sub_ranges[2]>0.0
                models[Threads.threadid()].alert_pars.p_test_sub = rand(Uniform(ensemble.p_test_sub_ranges...))
            else
                models[Threads.threadid()].alert_pars.p_test_sub = 0.0
            end

            models[Threads.threadid()].t_onset_to_isol = rand(Uniform(ensemble.t_isol_ranges[1]...))
            models[Threads.threadid()].alert_pars.t_onset_to_isol = min(rand(Uniform(ensemble.t_isol_ranges[2]...)), models[Threads.threadid()].t_onset_to_isol)

            models[Threads.threadid()].alert_pars.R_scaling = rand(Uniform(ensemble.alert_level_scaling_range...))
            models[Threads.threadid()].alert_pars.num_detected_before_alert = num_detected_before_alert*1
            models[Threads.threadid()].alert_pars.alert_level_scaling_speed = rand(Uniform(ensemble.alert_level_speed_range...))

            for key in keys(ensemble.extra_alert_event)
                if key == :k
                    models[Threads.threadid()].reproduction_k = ensemble.extra_alert_event[key] * 1.0
                elseif key == :contact
                    models[Threads.threadid()].alert_pars.p_contact = rand(Uniform(ensemble.extra_alert_event[key][:p_contact]...))
                    models[Threads.threadid()].alert_pars.p_contact_time = rand(Uniform(ensemble.extra_alert_event[key][:time]...))
                else
                    alertIncrease[i] = rand(ensemble.extra_alert_event[key][:R_scaling_increase])
                    push!(models[Threads.threadid()].alert_pars.extraALevents, AlertEvent(ensemble.extra_alert_event[key][:t_relative], models[Threads.threadid()].alert_pars.R_scaling * alertIncrease[i]))
                end
            end

            if USING__ALERT__MAX__VAL
                if modelReff(models[Threads.threadid()]) < ALERT__REFF__MAX__VAL
                    model_Reff_accepted[Threads.threadid()] = true
                end
            else
                model_Reff_accepted[Threads.threadid()] = true
            end

        end
        model_array[i] = deepcopy(models[Threads.threadid()])

        # next React branching process with alert level
        population_dfs[Threads.threadid()] = initDataframe(models[Threads.threadid()]);
        t, state_totals_all, num_cases = nextReact_branch!(population_dfs[Threads.threadid()], models[Threads.threadid()])


        # clean duplicate values of t which occur on the first recovery time.
        # breaks interpolation otherwise
        if initCases[Threads.threadid()] > 1
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])
        end

        firstDetectIndex = findfirst(state_totals_all[:,4].==num_detected_before_alert)

        if !isnothing(firstDetectIndex) && firstDetectIndex != length(state_totals_all[:,4])
            t_first_detect = t[firstDetectIndex]

            if detection_tspan[1] < t_first_detect && t_first_detect < detection_tspan[end]

                detectIndexes = findall(diff(state_totals_all[:,4]).==1) .+ 1
                undetectedIndexes = setdiff(collect(1:length(t)), detectIndexes)
                state_totals_all_new = state_totals_all[undetectedIndexes, 1:3]
                # tnew = t[undetectedIndexes] .- t_first_detect
                tnew = t .- t_first_detect

                IDetect = state_totals_all[detectIndexes, 4]
                tnew_detect = t[detectIndexes] .- t_first_detect


                StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, tnew, times)

                IDetect_tStep[:,i] = singleLinearSpline(state_totals_all[:, 4], tnew, times)
                dailyICases[:,i], dailyRCases[:,i], dailyDetectCases[:,i] = multipleLinearSplines(state_totals_all[:, 2:4], tnew, timesDaily)

                timeToHitAlertLower = t_first_detect+models[Threads.threadid()].alert_pars.alert_level_scaling_speed+1

                timeAfterHitAlertAllow = models[Threads.threadid()].t_max

                if CHANGING__ALERT__LEVELS
                    timeAfterHitAlertAllow = min(timeToHitAlertLower+models[Threads.threadid()].recovery_time+10+models[Threads.threadid()].alert_pars.alert_level_scaling_speed,  models[Threads.threadid()].t_max)
                end

                highAlert_df = filter(row -> row.parentID!=0 &&
                    row.time_infected > timeToHitAlertLower &&
                    row.active===false,
                    population_dfs[Threads.threadid()][1:num_cases, :], view=true)

                meanOffspring[i] = mean(highAlert_df.num_offspring)
                # meanOffspring[i] = nrow(highAlert_df)
                # meanRNumber[i] = mean(inactive_df.reproduction_number)


            else
                establishedSims[i] = false
            end
        else
            establishedSims[i] = false
        end

        if sum(isnan.(dailyDetectCases[:,i])) > 0 || sum(isnan.(IDetect_tStep[:,i])) > 0
            establishedSims[i] = false

            # dailyDetectCases = removeNaNs(dailyDetectCases, 2.0)
            # IDetect_tStep = removeNaNs(IDetect_tStep, 2.0)

            # println(state_totals_all[:,4])
            # println(dailyDetectCases[1,i])
            # println(t[firstDetectIndex])
            # println(state_totals_all[firstDetectIndex, 4])

            # println(findall(isnan.(dailyDetectCases[:,i])))
        end

        next!(p)
    end

    # dailyDetectCases = removeNaNs(dailyDetectCases, 2.0)
    # IDetect_tStep = removeNaNs(IDetect_tStep, 2.0)

    # dailyDetectCases = diff(hcat(convert.(Int64, zeros(length(timesDaily))), dailyDetectCases),dims=1)
    # dailyDetectCases = diff(vcat(transpose(zeros(numSims)), dailyDetectCases),dims=1)
    dailyDetectCases = diff(dailyDetectCases,dims=1)
    dailyTotalCases = dailyICases .+ dailyRCases
    dailyTotalCases = diff(dailyTotalCases, dims=1)

    IcumCases = ItStep .+ RtStep

    indexesToKeep = []
    sizehint!(indexesToKeep, numSims)

    for i in 2:length(establishedSims)
        if establishedSims[i]
            # xnew = hcat(xnew, x[:,i])
            push!(indexesToKeep, i)
        end
    end

    println("Kept $(length(indexesToKeep)) Sims or $(length(indexesToKeep)/numSims*100)% of Sims")

    # observedIDetect = [1, 10, 20, 30, 51]
    observedIDetect, tDetect = observedCases()

    return IcumCases[:,indexesToKeep], IDetect_tStep[:,indexesToKeep], observedIDetect, tDetect, dailyDetectCases[:,indexesToKeep], dailyTotalCases[:,indexesToKeep], model_array[indexesToKeep], meanOffspring[indexesToKeep], alertIncrease[indexesToKeep]
end

function newStatsTxt(newdir)
    io_output = open(newdir*"/outputStatistics.tex", "w")
    io_model = open(newdir*"/modelStatistics.txt", "w")
    # close(io)
    return io_output, io_model
end

function casesAtDetection(detectedCases, cumulativeCases, io) # cases are already filtered
    # detectedCases, cumulativeCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_26Aug.csv", true)[2:3]
    #
    # filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes.csv", header=false))
    # filterVector = convert.(Int64, filter_df.Column1)
    # detectedCases = detectedCases[:,filterVector]
    # cumulativeCases = cumulativeCases[:,filterVector]
    println(io, "Cumulative cases at detection")
    println(io, "Median cumulative cases of $(round(median(cumulativeCases, dims=2)[1]))")
    println(io, "IQR of [$(round(quantile2D(cumulativeCases, 0.25)[1],sigdigits=3)), $(round(quantile2D(cumulativeCases, 0.75)[1],sigdigits=3))]")

    # println(io, "Detected and cumulative cases at detection")
    # println(io, "Median detected cases of $(median(detectedCases, dims=2)[1])")
    # println(io, "IQR of [$(round(quantile2D(detectedCases, 0.25)[1],sigdigits=2)), $(round(quantile2D(detectedCases, 0.75)[1],sigdigits=2))]")
    # println(io, )

    # cumulativeCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_26Aug.csv", true)[3]
    println(io, "Detected and cumulative cases at $(Dates.format(DAY__ZERO__DATE + Dates.Day(70), "u d"))")
    println(io, "Median cumulative cases of $(round(median(cumulativeCases, dims=2)[70]))")
    println(io, "IQR of [$(round(quantile2D(cumulativeCases, 0.25)[70],sigdigits=3)), $(round(quantile2D(cumulativeCases, 0.75)[70],sigdigits=3))]")


    # println(io, "Detected and cumulative cases at $(Dates.format(DAY__ZERO__DATE + Dates.Day(70), "u d"))")
    println(io, "Median detected cases of $(round(median(detectedCases, dims=2)[70]))")
    println(io, "IQR of [$(round(quantile2D(detectedCases, 0.25)[70],sigdigits=3)), $(round(quantile2D(detectedCases, 0.75)[70],sigdigits=3))]")

    # println(io, median(cumulativeCases, dims=2)[70])
    # println(io, quantile2D(cumulativeCases, 0.25)[70])
    # println(io, quantile2D(cumulativeCases, 0.75)[70])

    # println(io, median(detectedCases, dims=2)[70])
    # println(io, quantile2D(detectedCases, 0.25)[70])
    # println(io, quantile2D(detectedCases, 0.75)[70])
    return io
end

function augustOutbreakPostProcess(processRange, Display=true, save=false)

    observedIDetect, tDetect = observedCases()

    println("Process #1: Effect of Alert Level Reff Decrease Speed on Daily Cases: August 2021 Sim using August 2020 Fit")
    if 1 in processRange

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Varying Alert Level Reff Decrease Speed"
        outputFileName = "./August2021Outbreak/VaryAlert_speed/DailyCaseNumbersAfterDetection"

        timesDaily, highAlertSpeedDailyDetect = reloadCSV("August2021Outbreak/CSVOutputs/HighAlert/BP2021fit_dailycases.csv", false)[1:2]
        lowAlertSpeedDailyDetect = reloadCSV("August2021Outbreak/CSVOutputs/LowAlert/BP2021fit_dailycases.csv", false)[2]
        modPDailyDetect = reloadCSV("August2021Outbreak/CSVOutputs/BP2021fit_dailycases.csv", false)[2]

        plotDailyCasesMultOutbreak(highAlertSpeedDailyDetect, modPDailyDetect, lowAlertSpeedDailyDetect, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, true)

        numCasesRange = 80:-10:10
        daysSinceRange = 30:5:70

        caseNumbersBeginToDrop(highAlertSpeedDailyDetect)
        probabilities = caseNumbersBeginToDrop(modPDailyDetect, numCasesRange, daysSinceRange)
        # println(probabilities)
        caseNumbersBeginToDrop(lowAlertSpeedDailyDetect)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = "./August2021Outbreak/ProbDaysSinceDetection_August2021Base"
        probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)
    end

    println("Process #2: August 2021 Model Ensemble Conditioned on case data to 10 days using Ccandu")
    if 2 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleConditioned26Aug")
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"
        io_output, io_model = newStatsTxt(newdir)
        observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210]#, 278, 348, 430, 513, 566, 615, 690, 739, 767, 787, 807, 827, 848]
        tDetect = collect(0:length(observedIDetect)-1)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 279]
        # tDetect = collect(0:length(observedIDetect)-1)

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_26Aug_old.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_26Aug_old.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_26Aug_old.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        numCasesRange = 80:-10:0
        daysSinceRange = 10:5:40

        t_current = 9
        probabilities = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)


        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)


        numCasesRange = 80:-4:10
        daysSinceRange = 10:5:40

        probabilities = caseNumbersBeginToDrop(dailyDetectedCases, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_cumulativecases_26Aug.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = "./August2020OutbreakFit/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        close(io_output); close(io_model)
    end

    println("Process #2.1: August 2021 Model Ensemble Conditioned on case data to 14 days using Ccandu")
    if 2.1 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleConditioned2Sep")
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_26Aug.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_26Aug.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_26Aug.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)



        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)


        # estimated hospitilisations ###########################################
        title = "Estimated cumulative hospitalisations"
        outputFileName = newdir*"/cumulativeHospitalisations_7_5%"
        plotHospEstimate(timesDaily[1:61], cumulativeDetectedCases[1:61,:], 0.075, title, outputFileName, Display, save)


        dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        title = "Estimated hospitalised cases"
        outputFileName = newdir*"/dailyInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)

        title = "Estimated critically hospitalised cases"
        outputFileName = newdir*"/dailyCritInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_cumulativecases_26Aug.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.2: August 2021 Model Ensemble Created 2 Sep, conditioned on 15 days using Ccandu")
    if 2.2 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated2Sep")
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_2Sep.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_2Sep.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_created2Sep_3Sep.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)


        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitilisations ###########################################
        dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        title = "Estimated hospitalised cases"
        outputFileName = newdir*"/dailyInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)

        title = "Estimated critically hospitalised cases"
        outputFileName = newdir*"/dailyCritInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)


        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_2Sep.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.3: August 2021 Model Ensemble Created 2 Sep, conditioned on daily using Ccandu")
    if 2.3 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated2Sep",true)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_2Sep.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_2Sep.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_created2Sep_3SepDaily.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)


        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitilisations ###########################################
        dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        title = "Estimated hospitalised cases"
        outputFileName = newdir*"/dailyInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)

        title = "Estimated critically hospitalised cases"
        outputFileName = newdir*"/dailyCritInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_2Sep.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.4: August 2021 Model Ensemble Created 3 Sep, conditioned on daily using Ccandu")
    if 2.4 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated3Sep",true)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_3Sep.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_3Sep.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_created3Sep_3SepDaily.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitilisations ###########################################
        # dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        # title = "Estimated hospitalised cases"
        # outputFileName = newdir*"/dailyInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)
        #
        # title = "Estimated critically hospitalised cases"
        # outputFileName = newdir*"/dailyCritInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_3Sep.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.5: August 2021 Model Ensemble Created 5 Sep, conditioned on daily using Ccandu") # current
    if 2.5 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated5Sep",true)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_created5Sep_8Sep.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_created5Sep_8Sep.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_config_8_9Sep.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitalisations ###########################################
        dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        title = "Estimated hospitalised cases"
        outputFileName = newdir*"/dailyInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)

        title = "Estimated critically hospitalised cases"
        outputFileName = newdir*"/dailyCritInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_created5Sep_8Sep.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        medReff = round(median(models_df[filterVector, :R_eff_after_al]),sigdigits=2)
        lowerQReff = round(quantile(models_df[filterVector,:R_eff_after_al], 0.25),sigdigits=2)
        upperQReff = round(quantile(models_df[filterVector,:R_eff_after_al], 0.75),sigdigits=2)
        print(io_model, "The median value of \$R_\\text{eff}\$ is $(medReff) ")
        println(io_model, "with IQR between [$lowerQReff, $upperQReff]" )

        println(io_model, "For 1.5x is $(medReff*1.5) and IQR [$(1.5*lowerQReff), $(1.5*upperQReff)] ")
        println(io_model, "For 2.0x is $(medReff*2) and IQR [$(2*lowerQReff), $(2*upperQReff)] ")
        println(io_model, "For 3.0x is $(medReff*3) and IQR [$(3*lowerQReff), $(3*upperQReff)] ")


        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.51: August 2021 Model Ensemble Created 5 Sep, ensemble run 8 Sep with AL3 change") # current
    if 2.51 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated5Sep/AlertChange",true)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_created5Sep_8SepAL3.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_created5Sep_8SepAL3.csv", true)[2:3]
        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_created5Sep_8SepAL3.csv")

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_config_9_8SepAL3.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        alertIncrease_vector = models_df[filterVector, :alertIncrease]
        Reff_vector = models_df[filterVector, :R_eff_after_al] .* alertIncrease_vector

        ####################################################################
        # load base-line and append

        timesDaily_base, dailyDetectedCases_base = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_created5Sep_8Sep.csv", false)[1:2]
        cumulativeDetectedCases_base, cumulativeTotalCases_base = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_created5Sep_8Sep.csv", true)[2:3]
        models_df_base = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_created5Sep_8Sep.csv")

        # cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        # cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df_base = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_config_8_8Sep.csv", header=false))
        filterVector_base = convert.(Int64, filter_df_base.Column1)
        dailyDetectedCases_base = dailyDetectedCases_base[:,filterVector_base]
        cumulativeDetectedCases_base = cumulativeDetectedCases_base[:,filterVector_base]
        cumulativeTotalCases_base = cumulativeTotalCases_base[:,filterVector_base]

        dailyDetectedCases = hcat(dailyDetectedCases, dailyDetectedCases_base[1:length(timesDaily), :])
        alertIncrease_vector = vcat(alertIncrease_vector, [1.0 for _ in 1:length(filterVector_base)])
        Reff_vector = vcat(Reff_vector, models_df_base[filterVector_base, :R_eff_after_al])
        #######################################################################

        io_output = caseChangeAfterAlertChange(dailyDetectedCases, io_output, alertIncrease_vector)

        alertIncreases = unique(alertIncrease_vector)

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        # title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        # outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        # plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "Daily Case Numbers After Detection, Changing Alert Level Effect on Reff"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection_AL3"
        plotDailyCasesMultOutbreak(dailyDetectedCases[:, findall(alertIncrease_vector .== 3.0)], dailyDetectedCases[:, findall(alertIncrease_vector .== 2.0)],
            dailyDetectedCases[:, findall(alertIncrease_vector .== 1.5)], dailyDetectedCases[:, findall(alertIncrease_vector .== 1.0)], timesDaily, observedIDetect, tDetect, title, outputFileName, false, true, true)


        # title = "August 2021 Outbreak Conditioned Model Ensemble"
        # outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"
        #
        # # write stats to file
        # # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)
        #
        # t_current = daysSinceZero()
        # plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
        #     observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
        #     true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitalisations ###########################################
        # dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        # title = "Estimated hospitalised cases"
        # outputFileName = newdir*"/dailyInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)
        #
        # title = "Estimated critically hospitalised cases"
        # outputFileName = newdir*"/dailyCritInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        # numCasesRange = 90:-1:0
        # daysSinceRange = 5:1:40
        #
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)
        #
        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)
        #
        # value = 10
        # io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        # # HAVE TO REDO FOR THIS SCENARIO
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_distplot"
        # plotReff(models_df, filterVector[1:end-length(filterVector_base)], io_output, title, outputFileName, Display, save)

        alertString_vector = ["" for _ in 1:length(alertIncrease_vector)]
        for i in 1:length(alertIncrease_vector)
            if alertIncrease_vector[i] == 3.0
                alertString_vector[i] = "High Increase ($(3)x)"
            elseif alertIncrease_vector[i] == 2.0
                alertString_vector[i] = "Moderate Increase ($(2)x)"
            elseif alertIncrease_vector[i] == 1.5
                alertString_vector[i] = "Low Increase ($(1.5)x)"
            else
                alertString_vector[i] = "No Change"
            end
        end

        # ["$(i)x increase" for i in alertIncrease_vector]

        plotReffAL3(Reff_vector, alertString_vector, ["High Increase ($(3)x)", "Moderate Increase ($(2)x)", "Low Increase ($(1.5)x)", "No Change"], title, outputFileName, true, true)

        medReff = round(median(models_df_base[filterVector_base, :R_eff_after_al]),sigdigits=2)
        lowerQReff = round(quantile(models_df_base[filterVector_base,:R_eff_after_al], 0.25),sigdigits=2)
        upperQReff = round(quantile(models_df_base[filterVector_base,:R_eff_after_al], 0.75),sigdigits=2)
        print(io_model, "The median value of \$R_\\text{eff}\$ is $(medReff) ")
        println(io_model, "with IQR between [$lowerQReff, $upperQReff]" )

        println(io_model, "For 1.5x is $(medReff*1.5) and IQR [$(1.5*lowerQReff), $(1.5*upperQReff)] ")
        println(io_model, "For 2.0x is $(medReff*2) and IQR [$(2*lowerQReff), $(2*upperQReff)] ")
        println(io_model, "For 3.0x is $(medReff*3) and IQR [$(3*lowerQReff), $(3*upperQReff)] ")

        println(io_output)

        for i in sort(alertIncreases)

            println(io_output, "\nFor $(i)x increase:")
            io_output = caseNumbersHitValue(dailyDetectedCases[:, findall(alertIncrease_vector .== i)], io_output, 0.5)

        end


        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.52: August 2021 Model Ensemble Created 5 Sep, conditioned on cumulative using ABC") # current
    if 2.52 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated5Sep/ABC",false)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_created5Sep_8Sep.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_created5Sep_8Sep.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_config_8_10Sep_ABC.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitalisations ###########################################
        dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        title = "Estimated hospitalised cases"
        outputFileName = newdir*"/dailyInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)

        title = "Estimated critically hospitalised cases"
        outputFileName = newdir*"/dailyCritInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_created5Sep_8Sep.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        medReff = round(median(models_df[filterVector, :R_eff_after_al]),sigdigits=2)
        lowerQReff = round(quantile(models_df[filterVector,:R_eff_after_al], 0.25),sigdigits=2)
        upperQReff = round(quantile(models_df[filterVector,:R_eff_after_al], 0.75),sigdigits=2)
        print(io_model, "The median value of \$R_\\text{eff}\$ is $(medReff) ")
        println(io_model, "with IQR between [$lowerQReff, $upperQReff]" )

        println(io_model, "For 1.5x is $(medReff*1.5) and IQR [$(1.5*lowerQReff), $(1.5*upperQReff)] ")
        println(io_model, "For 2.0x is $(medReff*2) and IQR [$(2*lowerQReff), $(2*upperQReff)] ")
        println(io_model, "For 3.0x is $(medReff*3) and IQR [$(3*lowerQReff), $(3*upperQReff)] ")


        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.55: August 2021 Model Ensemble Created 5 Sep, conditioned on daily using Ccandu for August 26th")
    if 2.55 in processRange

        observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210]#, 278, 348, 430, 513, 566, 615, 690, 739, 767, 787, 807, 827, 848]
        tDetect = collect(0:length(observedIDetect)-1)

        newdir = "./August2021Outbreak/EnsembleCreated5Sep/Updated26Aug_estimates"
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_created5Sep_7Sep.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_created5Sep_7Sep.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_created5Sep_26Aug.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = 9
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitalisations ###########################################
        dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        title = "Estimated hospitalised cases"
        outputFileName = newdir*"/dailyInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)

        title = "Estimated critically hospitalised cases"
        outputFileName = newdir*"/dailyCritInHospital"
        plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_created5Sep_7Sep.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.6: August 2021 Model Ensemble Created 6 Sep (contact tracing), conditioned on daily using Ccandu")
    if 2.6 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated6Sep_contact",true)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_6Sep_contact.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_6Sep_contact.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_created6Sep_6Sepcontact.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitilisations ###########################################
        # dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        # title = "Estimated hospitalised cases"
        # outputFileName = newdir*"/dailyInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)
        #
        # title = "Estimated critically hospitalised cases"
        # outputFileName = newdir*"/dailyCritInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_6Sep_contact.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.7: August 2021 Model Ensemble Created 7(8) Sep (contact tracing), conditioned on daily using Ccandu")
    if 2.7 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated7Sep_contact",true)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_7Sep_contact.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_7Sep_contact.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_config_7_9Sep.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitilisations ###########################################
        # dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        # title = "Estimated hospitalised cases"
        # outputFileName = newdir*"/dailyInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)
        #
        # title = "Estimated critically hospitalised cases"
        # outputFileName = newdir*"/dailyCritInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_7Sep_contact.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        medReff = round(median(models_df[filterVector, :R_eff_after_al]),sigdigits=2)
        lowerQReff = round(quantile(models_df[filterVector,:R_eff_after_al], 0.25),sigdigits=2)
        upperQReff = round(quantile(models_df[filterVector,:R_eff_after_al], 0.75),sigdigits=2)
        print(io_model, "The median value of \$R_\\text{eff}\$ is $(medReff) ")
        println(io_model, "with IQR between [$lowerQReff, $upperQReff]" )

        println(io_model, "For 1.5x is $(medReff*1.5) and IQR [$(1.5*lowerQReff), $(1.5*upperQReff)] ")
        println(io_model, "For 2.0x is $(medReff*2) and IQR [$(2*lowerQReff), $(2*upperQReff)] ")
        println(io_model, "For 3.0x is $(medReff*3) and IQR [$(3*lowerQReff), $(3*upperQReff)] ")




        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.71: August 2021 Model Ensemble Created 7(8) Sep (contact tracing), conditioned on cumulative using ABC")
    if 2.71 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated10Sep_contact",true)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_10Sep_contact.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_10Sep_contact.csv", true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_config_10_10Sep_ABC.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitilisations ###########################################
        # dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        # title = "Estimated hospitalised cases"
        # outputFileName = newdir*"/dailyInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)
        #
        # title = "Estimated critically hospitalised cases"
        # outputFileName = newdir*"/dailyCritInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_10Sep_contact.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save, true)

        medReff = round(median(models_df[filterVector, :R_eff_after_al_estimate]),sigdigits=2)
        lowerQReff = round(quantile(models_df[filterVector,:R_eff_after_al_estimate], 0.25),sigdigits=2)
        upperQReff = round(quantile(models_df[filterVector,:R_eff_after_al_estimate], 0.75),sigdigits=2)
        print(io_model, "The median value of \$R_\\text{eff}\$ is $(medReff) ")
        println(io_model, "with IQR between [$lowerQReff, $upperQReff]" )

        println(io_model, "For 1.5x is $(medReff*1.5) and IQR [$(1.5*lowerQReff), $(1.5*upperQReff)] ")
        println(io_model, "For 2.0x is $(medReff*2) and IQR [$(2*lowerQReff), $(2*upperQReff)] ")
        println(io_model, "For 3.0x is $(medReff*3) and IQR [$(3*lowerQReff), $(3*upperQReff)] ")




        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #2.72: August 2021 Model Ensemble Created 11 Sep (contact tracing) [transpose], conditioned on cumulative using ABC")
    if 2.72 in processRange

        newdir = createNewDir("./August2021Outbreak/EnsembleCreated11Sep_contact",true)
        # newdir = "./August2021Outbreak/2021_8_26_Aug/"

        io_output, io_model = newStatsTxt(newdir)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566]
        # tDetect = collect(0:length(observedIDetect)-1)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_11Sep_contact_transpose.csv", false, true)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_11Sep_contact_transpose.csv", true, true)[2:3]

        cumulativeDetectedCases_ensemble = cumulativeDetectedCases .* 1.0
        cumulativeTotalCases_ensemble = cumulativeTotalCases .* 1.0

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes_ensemble_config_11_11Sep_ABC.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetection"

        # write stats to file
        # io_output = casesAtDetection(cumulativeDetectedCases, cumulativeTotalCases, io_output)

        t_current = daysSinceZero()
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily,
            observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true,
            true, cumulativeDetectedCases_ensemble, cumulativeTotalCases_ensemble)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

        # estimated hospitilisations ###########################################
        # dailyHosp, dailyCrit = simulateHospDaily(timesDaily, dailyDetectedCases, 0.1, 0.17, 0.16)
        # title = "Estimated hospitalised cases"
        # outputFileName = newdir*"/dailyInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyHosp[1:61,:], title, outputFileName, Display, save)
        #
        # title = "Estimated critically hospitalised cases"
        # outputFileName = newdir*"/dailyCritInHospital"
        # plotHospEstimateDaily(timesDaily[1:61], dailyCrit[1:61,:], title, outputFileName, Display, save)
        ########################################################################

        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        value = 10
        io_output = caseNumbersHitValue(dailyDetectedCases, io_output, value)

        models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_11Sep_contact.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save, true)

        medReff = round(median(models_df[filterVector, :R_eff_after_al_estimate]),sigdigits=2)
        lowerQReff = round(quantile(models_df[filterVector,:R_eff_after_al_estimate], 0.25),sigdigits=2)
        upperQReff = round(quantile(models_df[filterVector,:R_eff_after_al_estimate], 0.75),sigdigits=2)
        print(io_model, "The median value of \$R_\\text{eff}\$ is $(medReff) ")
        println(io_model, "with IQR between [$lowerQReff, $upperQReff]" )

        println(io_model, "For 1.5x is $(medReff*1.5) and IQR [$(1.5*lowerQReff), $(1.5*upperQReff)] ")
        println(io_model, "For 2.0x is $(medReff*2) and IQR [$(2*lowerQReff), $(2*upperQReff)] ")
        println(io_model, "For 3.0x is $(medReff*3) and IQR [$(3*lowerQReff), $(3*upperQReff)] ")




        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #3: August 2021 Orig Fit for Aug 25")
    if 3 in processRange

        observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 279]
        tDetect = collect(0:length(observedIDetect)-1)

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021fit_dailycases.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2021Outbreak/CSVOutputs/BP2021fit_cumulativecases.csv", true)[2:3]

        # Ccandu filtering
        # filter_df = DataFrame(CSV.File("./August2021Outbreak/CSVOutputs/indexes.csv", header=false))
        # filterVector = filter_df.Column1
        # dailyDetectedCases = dailyDetectedCases[:,filterVector]
        # cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        # cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        numCasesRange = 80:-10:10
        daysSinceRange = 10:10:70

        probabilities = caseNumbersBeginToDrop(dailyDetectedCases, numCasesRange, daysSinceRange)

        t_current = 9

        title = "August 2021 Outbreak Conditioned Model Ensemble"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbersAfterDetection2020Fit"
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true)

        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = "./August2021Outbreak/ProbDaysSinceDetection_August2021Ccandu25Aug"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)


        numCasesRange = 80:-4:10
        daysSinceRange = 10:5:70

        probabilities = caseNumbersBeginToDrop(dailyDetectedCases, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = "./August2021Outbreak/ProbDaysSinceDetection_August2021_2020Fit"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)
    end

    println("Process #4: August 2020 Model Ensemble Conditioned on case data using Ccandu")
    if 4 in processRange

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = "./August2020OutbreakFit/DailyCaseNumbersAfterDetectionCcandu26Aug"

        timesDaily, dailyDetectedCases = reloadCSV("August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_26Aug.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_26Aug.csv", true)[2:3]

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2020OutbreakFit/CSVOutputs/indexes20.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, false, Display, save, true, true)

        numCasesRange = 30:-5:10
        daysSinceRange = 10:10:60

        probabilities = caseNumbersBeginToDrop(dailyDetectedCases, numCasesRange, daysSinceRange)

        t_current = -1

        title = "August 2020 Outbreak Conditioned Model Ensemble"
        outputFileName = "./August2020OutbreakFit/EstimatedCaseNumbersAfterDetectionCcandu26Aug"
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, true, false)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = "./August2020OutbreakFit/ProbDaysSinceDetection_August2020Ccandu26Aug"
        probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)


        numCasesRange = 20:-2:0
        daysSinceRange = 10:5:60

        probabilities = caseNumbersBeginToDrop(dailyDetectedCases, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = "./August2020OutbreakFit/ProbDaysSinceDetection_August2020Ccandu26Aug_Contour"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        models_df = reloadCSVmodels("./August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases_asymptomatic+lowR.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = "./August2020OutbreakFit/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)
    end

    println("Process #4.1: August 2020 Model Ensemble with asymptomatic test + low R Conditioned on case data using Ccandu")
    if 4.1 in processRange

        newdir = createNewDir("./August2020OutbreakFit/Asymptomatic+LowerR/")

        io_output, io_model = newStatsTxt(newdir)

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/DailyCaseNumbersAfterDetectionCcandu"

        timesDaily, dailyDetectedCases = reloadCSV("August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_asymptomatic+lowR.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_asymptomatic+lowR.csv", true)[2:3]

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2020OutbreakFit/CSVOutputs/indexes_1Sep.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, false, Display, save, true, false)

        numCasesRange = 30:-5:10
        daysSinceRange = 10:10:60

        t_current = -1

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)


        title = "August 2020 Outbreak Conditioned Model Ensemble"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/EstimatedCaseNumbersAfterDetectionCcandu"
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, false, false)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/ProbDaysSinceDetection_August2020Ccandu"
        probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)


        numCasesRange = 20:-2:0
        daysSinceRange = 10:5:60

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/ProbDaysSinceDetection_August2020Ccandu26Aug_Contour"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        models_df = reloadCSVmodels("./August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases_asymptomatic+lowR.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/Reff_violinplot"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #4.2: August 2020 Model Ensemble with asymptomatic test + low R fully conditioned on case data using Ccandu")
    if 4.2 in processRange

        newdir = createNewDir("./August2020OutbreakFit/Asymptomatic+LowerR/")

        io_output, io_model = newStatsTxt(newdir)

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/DailyCaseNumbersAfterDetectionCcandu_all_data"

        timesDaily, dailyDetectedCases = reloadCSV("August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_asymptomatic+lowR.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_asymptomatic+lowR.csv", true)[2:3]

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2020OutbreakFit/CSVOutputs/indexes_1Sep_all_data.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, false, Display, save, true, false)

        numCasesRange = 30:-5:10
        daysSinceRange = 10:10:60

        t_current = -1

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)


        title = "August 2020 Outbreak Conditioned Model Ensemble"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/EstimatedCaseNumbersAfterDetectionCcandu_all_data"
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, false, false)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/ProbDaysSinceDetection_August2020Ccandu_all_data"
        probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)


        numCasesRange = 20:-2:0
        daysSinceRange = 10:5:60

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/ProbDaysSinceDetection_August2020Ccandu26Aug_Contour_all_data"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        models_df = reloadCSVmodels("./August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases_asymptomatic+lowR.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/Reff_violinplot_all_data"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #4.3: August 2020 Model Ensemble with asymptomatic test + low R fully conditioned on case data using Ccandu")
    if 4.3 in processRange

        newdir = createNewDir("./August2020OutbreakFit/MatchMike/")

        io_output, io_model = newStatsTxt(newdir)

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)

        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]

        timesDaily, dailyDetectedCases = reloadCSV("August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_MatchMike.csv", false)[1:2]
        cumulativeDetectedCases, cumulativeTotalCases = reloadCSV("August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_MatchMike.csv", true)[2:3]

        # Ccandu filtering
        filter_df = DataFrame(CSV.File("./August2020OutbreakFit/CSVOutputs/indexes_ensemble_6Sep_MatchMike.csv", header=false))
        filterVector = convert.(Int64, filter_df.Column1)
        dailyDetectedCases = dailyDetectedCases[:,filterVector]
        cumulativeDetectedCases = cumulativeDetectedCases[:,filterVector]
        cumulativeTotalCases = cumulativeTotalCases[:,filterVector]

        title = "Daily Case Numbers After Detection, Conditioned Model Ensemble"
        outputFileName = newdir*"/DailyCaseNumbersAfterDetectionCcandu_all_data"
        plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, false, Display, save, true, false)

        t_current = -1
        title = "August 2020 Outbreak Conditioned Model Ensemble"
        outputFileName = newdir*"/EstimatedCaseNumbersAfterDetectionCcandu_all_data"
        plotAndStatsOutbreak(cumulativeDetectedCases, cumulativeTotalCases, timesDaily, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1, true, false, false)


        # numCasesRange = 30:-5:0
        # daysSinceRange = 10:10:60
        #
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)
        #
        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2020Ccandu_all_data"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)


        numCasesRange = 20:-1:0
        daysSinceRange = 10:1:60

        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2020Ccandu26Aug_Contour_all_data"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        models_df = reloadCSVmodels("./August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases_MatchMike.csv")
        title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        outputFileName = newdir*"/Reff_violinplot_all_data"
        plotReff(models_df, filterVector, io_output, title, outputFileName, Display, save)

        # write model info to io
        close(io_output); close(io_model)
    end

    println("Process #5: BP Team ChCh Conditioned Ensemble Contour plot 2 Sep")
    if 5 in processRange

        newdir = "."
        io_output, io_model = newStatsTxt(newdir)

        timesDaily, dailyDetectedCases = reloadCSV("August2021Outbreak/CSVOutputs/bpteamforjoel.csv", false)[1:2]


        # tspan = (0.0,70.0)
        # timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, BPM team"
        outputFileName = "./DailyCaseNumbersAfterDetection_BPMteam"
        # plotDailyCasesOutbreak(dailyDetectedCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true, true)

        # numCasesRange = 80:-10:10
        # daysSinceRange = 10:5:40
        # probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, numCasesRange, daysSinceRange)



        # title = "Probability of less than N cases per day, x days after detection"
        # outputFileName = newdir*"/ProbDaysSinceDetection_August2021"
        # probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)


        numCasesRange = 90:-1:0
        daysSinceRange = 5:1:40

        t_current=-1
        probabilities, io_output = caseNumbersBeginToDrop(dailyDetectedCases, io_output, t_current, numCasesRange, daysSinceRange)

        title = "Probability of less than N cases per day, x days after detection"
        outputFileName = newdir*"/ProbDaysSinceDetection_August2021_BPMteam_2Sep_normalscale"
        probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save, true)

        close(io_output)

        # models_df = reloadCSVmodels("./August2021Outbreak/CSVOutputs/Models/BP2021ensemble_cumulativecases_26Aug.csv")
        # title = "Value of Reff pre and post Alert Level, conditioned and unconditioned"
        # outputFileName = "./August2020OutbreakFit/Reff_violinplot"
        # plotReff(models_df, filterVector, title, outputFileName, Display, save)

        # write model info to io
        close(io_model)
    end

    return nothing
end

function augustOutbreakSim(numSimsScaling::Union{Float64,Int64}, simRange, Display=true, save=true, CSVOutputRange=[])
    #=
    Estimation of the August 2021 Delta outbreak on 18 Aug.

    1 seed case, estimated to have been seeded 10 to 17 days ago.

    Estimate the cumulative number of infections from this time point.
    =#

    observedIDetect, tDetect = observedCases()

    println("Sim #1: Heterogeneous Reproduction Number")
    if 1 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,17.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(10000/numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        population_dfs = [initDataframe(models[1]) for i in 1:Threads.nthreads()];
        i = 1
        p = Progress(numSims,PROGRESS__METER__DT)

        meanOffspring = zeros(numSims)
        meanRNumber = zeros(numSims)

        establishedSims = BitArray(undef, numSims) .* false
        establishedSims .= true

        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^6, 5*10^3, [5*10^6-1,1,0]);
            models[Threads.threadid()].stochasticRi = true
            models[Threads.threadid()].reproduction_number = 5
            models[Threads.threadid()].p_test = 0.4
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_dfs[Threads.threadid()] = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_dfs[Threads.threadid()], models[Threads.threadid()])

            if num_cases < 3
                establishedSims[i] = false
            end

            # clean duplicate values of t which occur on the first recovery time
            # firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            # lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            # t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            # state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

            inactive_df = filter(row -> row.parentID!=0, population_dfs[Threads.threadid()][1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            meanRNumber[i] = mean(inactive_df.reproduction_number)

            next!(p)
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $(median(meanRNumber))")
        println("Finished Simulation in $time seconds")
        x = ItStep .+ RtStep

        # xnew = x[:,1]
        # sizehint!(xnew, size(x)...)
        indexesToKeep = []
        sizehint!(indexesToKeep, numSims)

        for i in 2:length(establishedSims)
            if establishedSims[i]
                # xnew = hcat(xnew, x[:,i])
                push!(indexesToKeep, i)
            end
        end

        title = "August 2021 Outbreak Cumulative Infection Estimation"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbers"
        plotCumulativeInfections(x[:,indexesToKeep], times, [10.0,13.0,15.0,17.0], title, outputFileName, Display, save, 0.1)
    end

    println("Sim #2: Homogeneous Reproduction Number")
    if 2 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021 - no Heteregeneity
        # time span to sim on
        tspan = (0.0,17.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(10000/numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        establishedSims = BitArray(undef, numSims) .* false
        establishedSims .= true

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        population_dfs = [initDataframe(models[1]) for i in 1:Threads.nthreads()];
        i = 1
        p = Progress(numSims,PROGRESS__METER__DT)
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^6, 5*10^3, [5*10^6-1,1,0]);
            models[Threads.threadid()].stochasticRi = false
            models[Threads.threadid()].reproduction_number = 5
            models[Threads.threadid()].p_test = 0.4
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            population_dfs[Threads.threadid()] = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_dfs[Threads.threadid()], models[Threads.threadid()])

            if num_cases < 3
                establishedSims[i] = false
            end

            # clean duplicate values of t which occur on the first recovery time
            # firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            # lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            # t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            # state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
            next!(p)
        end

        println("Finished Simulation in $time seconds")
        x = ItStep .+ RtStep

        indexesToKeep = []
        sizehint!(indexesToKeep, numSims)

        # xnew = x[:,1]
        # sizehint!(xnew, size(x)...)

        for i in 1:length(establishedSims)
            if establishedSims[i]
                # xnew = hcat(xnew, x[:,i])
                push!(indexesToKeep, i)
            end
        end

        title = "August 2021 Outbreak Cumulative Infection Estimation"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbersHomogeneous"
        plotCumulativeInfections(x[:,indexesToKeep], times, [10.0,13.0,15.0,17.0], title, outputFileName, Display, save, 0.1)


        ################################################################################
    end

    println("Sim #3: Affect of p_test on infection curves post first detection, August Outbreak Sim")
    if 3 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,60.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (10,17)

        maxCases=20*10^3
        p_test=[0.1,0.8]
        R_number=6
        R_alert_scaling=0.2
        t_onset_to_isol=[2.2,1.0]
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
            maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol)

        t_current = 4

        # title = "August 2021 Outbreak, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, true, true, true)

        title = "August 2021 Outbreak, Estimated Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbersAfterDetection"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
    end

    println("Sim #4: Fitting on August 2020 Outbreak")
    if 4 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,60.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(5000/numSimsScaling))

        detection_tspan = (13,18)

        initCases=1
        maxCases=5*10^3
        p_test=[0.1,0.9]
        R_number=4
        R_alert_scaling=0.25 #= was 0.25, bumped to 0.3 when made linear reduction
                                in R_scaling actually work.=#
        t_onset_to_isol=[2.2,0.8]
        cumulativeCases, confirmedCases, observedIDetect, tDetect, dailyDetectCases =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
            maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases)

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        t_current = 0 # don't put date on plot

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2020 Outbreak, Daily Case Numbers After Detection"
        outputFileName = "./August2020OutbreakFit/DailyCaseNumbersAfterDetection"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, false, true, true)

        title = "August 2020 Outbreak, Estimated Case Numbers After Detection"
        outputFileName = "./August2020OutbreakFit/EstimatedCaseNumbersAfterDetection"
        plotAndStatsOutbreak(confirmedCases, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
    end

    println("Sim #4.1: August 2020 Sim Model Ensemble for Ccandu fitting")
    if 4.1 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,60.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(40000/numSimsScaling))

        detection_tspan = (10,20) # increase from 6,14

        maxCases=5*10^3

        num_detected_before_alert=1

        # ensemble = Ensemble([[0.05,0.15],[0.6,1.0]],[0.0,0.0], [[1.0,3.0],[0.5,2.0]],[0.10,0.30],[2,10],[5,7], [1,2])
        ensemble = Ensemble([[0.05,0.1],[0.8,1.0]], [0.0,0.0], [[1.0,3.0],[0.5,1.5]],[0.20,0.30],[2,10],[3.5,4.5], [1,2], Dict())

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases, model_array = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = 0

        cumulativeCases = removeNaNs(cumulativeCases)

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2020 Ensemble, Daily Case Numbers After Detection"
        outputFileName = "./August2020OutbreakFit/DailyCaseNumbersAfterDetectionEnsemble26Aug"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, false)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2020OutbreakFit/EstimatedCaseNumbersAfterDetectionEnsemble26Aug"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 4.1 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_26Aug.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_26Aug.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            "August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases.csv"
            outputCSVmodels(model_array, outputFileName)
        end
    end

    println("Sim #4.2: August 2020 Sim Model Ensemble for Ccandu fitting. Detection of asymptomatic")
    if 4.2 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2020
        # time span to sim on
        tspan = (0.0,60.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(40000/numSimsScaling))

        detection_tspan = (10,20) # increase from 6,14

        maxCases=5*10^3

        num_detected_before_alert=1

        # ensemble = Ensemble([[0.05,0.15],[0.6,1.0]],[0.0,0.0], [[1.0,3.0],[0.5,2.0]],[0.10,0.30],[2,10],[5,7], [1,2])
        ensemble = Ensemble([[0.05,0.1],[0.6,0.8]], [0.6,0.8], [[1.0,3.0],[0.5,2.0]],[0.20,0.30],[2,10],[3.5,4.5], [1,2], Dict())

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases, model_array = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = 0

        cumulativeCases = removeNaNs(cumulativeCases)

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2020 Ensemble, Daily Case Numbers After Detection"
        outputFileName = "./August2020OutbreakFit/AddingAsymptomaticDetection/DailyCaseNumbersAfterDetectionEnsemble26Aug"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, false)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2020OutbreakFit/AddingAsymptomaticDetection/EstimatedCaseNumbersAfterDetectionEnsemble26Aug"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 4.2 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_asymptomatic.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_asymptomatic.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            "August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases_asymptomatic.csv"
            outputCSVmodels(model_array, outputFileName)
        end
    end

    println("Sim #4.3: August 2020 Sim Model Ensemble for Ccandu fitting. Detection of asymptomatic and more authentic Reff for reg covid")
    if 4.3 in simRange

        # better estimate of Reff for regular COVID is 2.3. Rather than current 3.3 (4 * 5/6).
        # centre Reff around 2.3 (R around 2.76)

        ################################################################################
        # Simulating Delta Outbreak 18 August 2020
        # time span to sim on
        tspan = (0.0,60.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(40000/numSimsScaling))

        detection_tspan = (0,59) # increase from 6,14

        maxCases=5*10^3

        num_detected_before_alert=1

        # ensemble = Ensemble([[0.05,0.15],[0.6,1.0]],[0.0,0.0], [[1.0,3.0],[0.5,2.0]],[0.10,0.30],[2,10],[5,7], [1,2])
        ensemble = Ensemble([[0.05,0.1],[0.6,0.8]], [0.6,0.8], [[1.0,3.0],[0.5,2.0]],[0.30,0.40],[2,10],[2.5,3.5], [1,2], Dict())

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases, model_array = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = 0

        cumulativeCases = removeNaNs(cumulativeCases)

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2020 Ensemble, Daily Case Numbers After Detection"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/DailyCaseNumbersAfterDetectionEnsemble26Aug"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, false)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR/EstimatedCaseNumbersAfterDetectionEnsemble26Aug"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 4.3 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_asymptomatic+lowR.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_asymptomatic+lowR.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            outputFileName = "August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases_asymptomatic+lowR.csv"
            outputCSVmodels(model_array, outputFileName)
        end
    end

    println("Sim #4.4: August 2020 Sim Model Ensemble for Ccandu fitting. Detection of asymptomatic and more authentic Reff for reg covid, Al3->AL2")
    if 4.4 in simRange

        # better estimate of Reff for regular COVID is 2.3. Rather than current 3.3 (4 * 5/6).
        # centre Reff around 2.3 (R around 2.76)

        ################################################################################
        # Simulating Delta Outbreak 18 August 2020
        # time span to sim on
        tspan = (0.0,60.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(40000/numSimsScaling))

        detection_tspan = (10,25) # increase from 6,14

        maxCases=5*10^3

        num_detected_before_alert=1

        # ensemble = Ensemble([[0.05,0.15],[0.6,1.0]],[0.0,0.0], [[1.0,3.0],[0.5,2.0]],[0.10,0.30],[2,10],[5,7], [1,2])
        ensemble = Ensemble([[0.05,0.1],[0.6,0.8]], [0.6,0.8], [[1.0,3.0],[0.5,2.0]],[0.30,0.40],[2,10],[2.5,3.5], [1,2], Dict(2=>Dict(:t_relative=>20, :R_scaling_increase=>[1.5])))

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases, model_array = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = 0

        cumulativeCases = removeNaNs(cumulativeCases)

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2020 Ensemble, Daily Case Numbers After Detection"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR+AL2/DailyCaseNumbersAfterDetectionEnsemble26Aug"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, false)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2020OutbreakFit/Asymptomatic+LowerR+AL2/EstimatedCaseNumbersAfterDetectionEnsemble26Aug"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 4.4 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_asymptomatic+lowR+AL2.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_asymptomatic+lowR+AL2.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            outputFileName = "August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases_asymptomatic+lowR+AL2.csv"
            outputCSVmodels(model_array, outputFileName)
        end
    end

    println("Sim #4.5: August 2020 Sim Model Ensemble for Ccandu fitting. Using similar parameters to Mike et. al. fitting on same data")
    if 4.5 in simRange

        # better estimate of Reff for regular COVID is 2.3. Rather than current 3.3 (4 * 5/6).
        # centre Reff around 2.3 (R around 2.76)

        ################################################################################
        # Simulating Delta Outbreak 18 August 2020
        # time span to sim on
        tspan = (0.0,60.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(15000/numSimsScaling))

        detection_tspan = (0,59) # increase from 6,14

        maxCases=5*10^3

        num_detected_before_alert=1

        # ensemble = Ensemble([[0.05,0.15],[0.6,1.0]],[0.0,0.0], [[1.0,3.0],[0.5,2.0]],[0.10,0.30],[2,10],[5,7], [1,2])
        ensemble = Ensemble([[0.05,0.15],[0.8,1.0]], [0.0,0.0], [[3.0,7.0],[0.5,4.0]],[0.20,0.40],[2,10],[2.5,3.8], [1,2], Dict(:k=>0.7))

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases, model_array = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = 0

        cumulativeCases = removeNaNs(cumulativeCases)

        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2020 Ensemble, Daily Case Numbers After Detection"
        outputFileName = "./August2020OutbreakFit/MatchMike/DailyCaseNumbersAfterDetectionEnsemble6Sep"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, false)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2020 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2020OutbreakFit/MatchMike/EstimatedCaseNumbersAfterDetectionEnsemble6Sep"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 4.5 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_dailycases_MatchMike.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2020OutbreakFit/CSVOutputs/BP2020ensemble_cumulativecases_MatchMike.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            outputFileName = "August2020OutbreakFit/CSVOutputs/Models/BP2020ensemble_cumulativecases_MatchMike.csv"
            outputCSVmodels(model_array, outputFileName)
        end
    end

    modPDailyDetect = []
    println("Sim #5: August 2021 Sim using August 2020 Fit")
    if 5 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,15) # increase from 6,14

        initCases=1

        maxCases=20*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=6
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.8] # increase from 2.2, 0.1
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases = baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 6

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, true, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak using August 2020 Fit \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbersAfterDetection2020Fit"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)

        if 5 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/BP2021fit_dailycases.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/BP2021fit_cumulativecases.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)
        end
    end

    println("Sim #5.1: August 2021 Sim Model Ensemble for Ccandu fitting")
    if 5.1 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(20000/numSimsScaling))

        detection_tspan = (6,20) # increase from 6,14

        maxCases=40*10^3

        num_detected_before_alert=1

        # ensemble = Ensemble([[0.05,0.15],[0.6,1.0]], [0.0,0.0], [[1.0,3.0],[0.5,2.0]],[0.10,0.30],[2,10],[5,7], [1,2])
        ensemble = Ensemble([[0.05,0.1],[0.8,1.0]], [0.0,0.0], [[1.0,3.0],[0.5,1.5]],[0.20,0.25],[2,10],[5.5,6.5], [1], Dict())

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases, model_array = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = 9

        cumulativeCases = removeNaNs(cumulativeCases)

        observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 279]
        tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetectionEnsemble26Aug"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbersAfterDetectionEnsemble26Aug"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 5.1 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_26Aug.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_26Aug.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            outputFileName = "August2021Outbreak/CSVOutputs/Models/BP2021ensemble_cumulativecases_26Aug.csv"
            outputCSVmodels(model_array, outputFileName)
        end
    end

    println("Sim #5.2: August 2021 Sim Model Ensemble for Ccandu fitting, Lower Rscaling")
    if 5.2 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(14000/numSimsScaling))

        detection_tspan = (6,20) # increase from 6,14

        maxCases=40*10^3

        num_detected_before_alert=1

        # ensemble = Ensemble([[0.05,0.15],[0.6,1.0]], [0.0,0.0], [[1.0,3.0],[0.5,2.0]],[0.10,0.30],[2,10],[5,7], [1,2])
        # ensemble = Ensemble([[0.05,0.1],[0.55,0.8]], [0.55,0.8], [[1.0,3.0],[0.5,1.5]],[0.20,0.25],[2,10],[5.5,6.5], [1,2], Dict())

        ensemble = Ensemble([[0.05,0.1],[0.8,1.0]], [0.0,0.0], [[1.0,3.0],[0.5,1.5]],[0.15,0.25],[2,10],[5.5,6.5], collect(1:4), Dict())

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases, model_array = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = daysSinceZero()

        cumulativeCases = removeNaNs(cumulativeCases)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 279]
        # tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/EnsembleCreated1Sep/DailyCaseNumbersAfterDetectionEnsemble2Sep"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2021Outbreak/EnsembleCreated1Sep/EstimatedCaseNumbersAfterDetectionEnsemble2Sep"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 5.2 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_2Sep.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_2Sep.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            outputFileName = "August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_2Sep.csv"
            outputCSVmodels(model_array, outputFileName)
        end
    end

    println("Sim #5.3: August 2021 Sim Model Ensemble for Ccandu fitting, Higher testing")
    if 5.3 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(14000/numSimsScaling))

        detection_tspan = (6,20) # increase from 6,14

        maxCases=25*10^3

        num_detected_before_alert=1

        # ensemble = Ensemble([[0.05,0.15],[0.6,1.0]], [0.0,0.0], [[1.0,3.0],[0.5,2.0]],[0.10,0.30],[2,10],[5,7], [1,2])
        # ensemble = Ensemble([[0.05,0.1],[0.55,0.8]], [0.55,0.8], [[1.0,3.0],[0.5,1.5]],[0.20,0.25],[2,10],[5.5,6.5], [1,2], Dict())

        ensemble = Ensemble([[0.05,0.1],[0.7,1.0]], [0.6,1.0], [[1.0,3.0],[2.5,4.0]],[0.1,0.25],[2,12],[5.5,6.5], collect(1:4), Dict())

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases, model_array = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = daysSinceZero()

        cumulativeCases = removeNaNs(cumulativeCases)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 279]
        # tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/EnsembleCreated1Sep/DailyCaseNumbersAfterDetectionEnsemble2Sep"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2021Outbreak/EnsembleCreated1Sep/EstimatedCaseNumbersAfterDetectionEnsemble2Sep"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 5.3 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_3Sep.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_3Sep.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            outputFileName = "August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_3Sep.csv"
            outputCSVmodels(model_array, outputFileName)
        end
    end

    println("Sim #5.4: August 2021 Sim Model Ensemble for Ccandu fitting, Even Lower Rscaling") # Currently the one being used
    if 5.4 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(14000/numSimsScaling))

        detection_tspan = (6,20) # increase from 6,14

        maxCases=50*10^3

        num_detected_before_alert=1

        ensemble = Ensemble([[0.05,0.1],[0.7,1.0]], [0.0,0.3], [[1.0,3.0],[0.5,1.5]],[0.01,0.5],[2,10],[5.5,6.5], collect(1:4), Dict()) #Used for Sep 5+6
        # ensemble = Ensemble([[0.05,0.1],[0.7,1.0]], [0.0,0.3], [[1.0,3.0],[0.5,1.5]],[0.01,0.33],[2,10],[5.5,6.5], collect(1:4), Dict()) #Used for Sep 7
        # ensemble = Ensemble([[0.05,0.1],[0.7,1.0]], [0.0,0.3], [[1.0,3.0],[0.5,1.5]],[0.01,0.45],[2,10],[5.5,6.5], collect(1:4), Dict()) #Used for Sep 8 + enforce maxReff in ensemble outbreak sim of 1.0

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases,
            model_array, meanOffspringAL = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = daysSinceZero()

        cumulativeCases = removeNaNs(cumulativeCases)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 279]
        # tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/EnsembleCreated5Sep/DailyCaseNumbersAfterDetectionEnsemble8Sep"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2021Outbreak/EnsembleCreated5Sep/EstimatedCaseNumbersAfterDetectionEnsemble8Sep"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 5.4 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_created5Sep_8Sep.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_created5Sep_8Sep.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            outputFileName = "August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_created5Sep_8Sep.csv"
            outputCSVmodels(model_array, outputFileName, meanOffspringAL)
        end
    end

    println("Sim #5.41: August 2021 Sim Model Ensemble for Ccandu fitting, Even Lower Rscaling AL3 [1.5x, 2x and 3x] Reff") # Currently the one being used
    if 5.41 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,60.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(14000/numSimsScaling))

        detection_tspan = (6,20) # increase from 6,14

        maxCases=90*10^3

        num_detected_before_alert=1

        ensemble = Ensemble([[0.05,0.1],[0.7,1.0]], [0.0,0.3], [[1.0,3.0],[0.5,1.5]],[0.01,0.5],[2,10],[5.5,6.5], collect(1:4),
            Dict(3=>Dict(:t_relative=>daysFromZeroToALChange()-1, :R_scaling_increase=>[1.5, 2, 3])))

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases,
            model_array, meanOffspringAL, alertIncrease = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = daysSinceZero()

        cumulativeCases = removeNaNs(cumulativeCases)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 279]
        # tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/EnsembleCreated5Sep/DailyCaseNumbersAfterDetectionEnsemble8SepAL3"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak Model Ensemble for Ccandu"
        outputFileName = "./August2021Outbreak/EnsembleCreated5Sep/EstimatedCaseNumbersAfterDetectionEnsemble8SepAL3"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        # output CSVs
        if 5.41 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_created5Sep_8SepAL3.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_created5Sep_8SepAL3.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)

            outputFileName = "August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_created5Sep_8SepAL3.csv"
            outputCSVmodels(model_array, outputFileName, meanOffspringAL, alertIncrease)
        end
    end

    println("Sim #5.5: August 2021 Sim Model Ensemble for Ccandu fitting, Even Lower Rscaling + Contact Tracing")
    if 5.5 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(100000/numSimsScaling))

        detection_tspan = (6,20) # increase from 6,14

        maxCases=20*10^3

        num_detected_before_alert=1

        # 6 Sep
        # ensemble = Ensemble([[0.05,0.15],[0.5,1.0]], [0.0,0.0], [[2.0,5.0],[0.5,4.0]],[0.01,0.3],[2,10],[5.5,6.5], collect(1:4), Dict(:contact=>Dict(:p_contact=>[0.0,0.7], :time=>[2.0,6.0])))

        # 7 Sep
        # ensemble = Ensemble([[0.05,0.15],[0.4,0.8]], [0.0,0.0], [[2.0,5.0],[1.0,4.0]],[0.01,0.4],[2,10],[5.5,6.5], collect(1:4), Dict(:contact=>Dict(:p_contact=>[0.5,0.9], :time=>[2.0,6.0])))

        # 11 Sep, 100,000 sims, tranposed CSV
        ensemble = Ensemble([[0.05,0.15],[0.3,0.8]], [0.0,0.0], [[2.0,5.0],[1.0,4.0]],[0.01,0.4],[2,10],[5.5,6.5], collect(1:4), Dict(:contact=>Dict(:p_contact=>[0.5,0.95], :time=>[2.0,6.0])))

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases,
            dailyTotalCases, model_array, meanOffspringAL = ensembleOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                ensemble, maxCases, num_detected_before_alert)

        t_current = daysSinceZero()

        cumulativeCases = removeNaNs(cumulativeCases)

        # observedIDetect = [1, 10, 20, 32, 51, 74, 107, 148, 210, 279]
        # tDetect = collect(0:length(observedIDetect)-1)


        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        if Display || save
            title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
            outputFileName = "./August2021Outbreak/EnsembleCreated6Sep_contact/DailyCaseNumbersAfterDetectionEnsemble10Sep"
            plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save, true)
            # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

            title = "August 2021 Outbreak Model Ensemble for Ccandu"
            outputFileName = "./August2021Outbreak/EnsembleCreated6Sep_contact/EstimatedCaseNumbersAfterDetectionEnsemble10Sep"
            plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)
        end

        # output CSVs
        if 5.5 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_11Sep_contact_transpose.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false, true)

            outputFileName = "August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_11Sep_contact_transpose.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true, true)

            outputFileName = "August2021Outbreak/CSVOutputs/Models/BP2021ensemble_models_11Sep_contact.csv"
            outputCSVmodels(model_array, outputFileName, meanOffspringAL)
        end
    end

    println("Sim #5.9: August 2021 Sim using August 2020 Fit") # original simulation params
    if 5.9 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,14)

        initCases=1

        maxCases=10*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=5
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.1]
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases = baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 4

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit2"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, true, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak using August 2020 Fit \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbersAfterDetection2020Fit2"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
    end

    highRPreDailyDetect = []
    println("Sim #6: High R_i: August 2021 Sim using August 2020 Fit")
    if 6 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,14)

        initCases=1

        maxCases=20*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=6
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.1]
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases  =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 4

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end
        highRPreDailyDetect = dailyDetectCases
        # title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, true, true, true)

        title = "August 2021 Outbreak using August 2020 Fit, High R0"# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryR_i/EstimatedCaseNumbersAfterDetection2020FitHighR"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
    end

    lowRPreDailyDetect = []
    println("Sim #7: Low R_i: August 2021 Sim using August 2020 Fit")
    if 7 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(20000/numSimsScaling))

        detection_tspan = (6,14)

        initCases=1

        maxCases=8*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=4
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.1]
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases  =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 4

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        lowRPreDailyDetect = dailyDetectCases
        # title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, Display, save, true)

        title = "August 2021 Outbreak using August 2020 Fit, Low R0"# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryR_i/EstimatedCaseNumbersAfterDetection2020FitLowR"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)
    end

    println("Sim #7.5: Effect of R Pre scaling on Daily Cases: August 2021 Sim using August 2020 Fit")
    if 5 in simRange && 6 in simRange && 7 in simRange

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Varying Pre Intervention Reff"
        outputFileName = "./August2021Outbreak/VaryR_i/DailyCaseNumbersAfterDetection"
        plotDailyCasesMultOutbreak(highRPreDailyDetect, modPDailyDetect, lowRPreDailyDetect, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save)
    end

    highPDailyDetect = []
    println("Sim #8: High P_test: August 2021 Sim using August 2020 Fit")
    if 8 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,14)

        initCases=1

        maxCases=10*10^3
        p_test=[0.1,1.0]#p_test=[0.1,0.8]
        R_number=5
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.1]
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 4

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        highPDailyDetect = dailyDetectCases

        # title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, Display, save, true)

        title = "August 2021 Outbreak using August 2020 Fit, High P_test"# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryP_test/EstimatedCaseNumbersAfterDetection2020FitHighP_test"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)
    end

    lowPDailyDetect = []
    println("Sim #9: Low P_test: August 2021 Sim using August 2020 Fit")
    if 9 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,14)

        initCases=1

        maxCases=11*10^3
        p_test=[0.1,0.6]#p_test=[0.1,0.8]
        R_number=5
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,1.0]
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases  =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 4

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        lowPDailyDetect = dailyDetectCases

        # title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, Display, save, true)

        title = "August 2021 Outbreak using August 2020 Fit, Low P_test"# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryP_test/EstimatedCaseNumbersAfterDetection2020FitLowP_test"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)
    end

    println("Sim #9.5: Effect of P_test scaling on Daily Cases: August 2021 Sim using August 2020 Fit")
    if 5 in simRange && 8 in simRange && 9 in simRange

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Varying Symptomatic Testing Rate"
        outputFileName = "./August2021Outbreak/VaryP_test/DailyCaseNumbersAfterDetection"
        plotDailyCasesMultOutbreak(highPDailyDetect, modPDailyDetect, lowPDailyDetect, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save)
    end

    highRPostDailyDetect = []
    println("Sim #10: High R_scaling: August 2021 Sim using August 2020 Fit")
    if 10 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,14)

        initCases=1

        maxCases=10*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=5
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.15
        t_onset_to_isol=[2.2,0.1]
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 4

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        highRPostDailyDetect = dailyDetectCases

        # title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, Display, save, true)

        title = "August 2021 Outbreak using August 2020 Fit, High Alert Level Reduction of R0 "# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryR_scaling/EstimatedCaseNumbersAfterDetection2020FitHighAlert_Reduct"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)
    end

    lowRPostDailyDetect = []
    println("Sim #11: Low R_scaling: August 2021 Sim using August 2020 Fit")
    if 11 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 0.1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,14)

        initCases=1

        maxCases=20*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=5
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.35
        t_onset_to_isol=[2.2,0.1]
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 4

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end
        lowRPostDailyDetect = dailyDetectCases

        # title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, Display, save, true)

        title = "August 2021 Outbreak using August 2020 Fit, Low Alert Level Reduction of R0 "# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryR_scaling/EstimatedCaseNumbersAfterDetection2020FitLowAlert_Reduct"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)
    end

    println("Sim #11.5: Effect of R Post scaling on Daily Cases: August 2021 Sim using August 2020 Fit")
    if 5 in simRange && 10 in simRange && 11 in simRange

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Varying Post Intervention Reff"
        outputFileName = "./August2021Outbreak/VaryR_scaling/DailyCaseNumbersAfterDetection"
        plotDailyCasesMultOutbreak(highRPostDailyDetect, modPDailyDetect, lowRPostDailyDetect, timesDaily, observedIDetect, tDetect, title, outputFileName, true, Display, save)
    end

    highAlertSpeedDailyDetect = []
    println("Sim #12: High Alert Level Reff Decrease Speed: August 2021 Sim using August 2020 Fit")
    if 12 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,15)

        initCases=1

        maxCases=14*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=6
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.8]
        num_detected_before_alert=3
        alert_scaling_speed = 2

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert, alert_scaling_speed)

        t_current = 6

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        highAlertSpeedDailyDetect = dailyDetectCases

        # title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, Display, save, true)

        title = "August 2021 Outbreak using August 2020 Fit, High Alert Level Reff Decrease Speed"# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryAlert_speed/EstimatedCaseNumbersAfterDetection2020FitHighAlert_speed"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        if 12 in CSVOutputRange

            timesDaily = [i for i=tspan[1]:1:tspan[end]]
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/HighAlert/BP2021fit_dailycases.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/HighAlert/BP2021fit_cumulativecases.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)
        end
    end

    lowAlertSpeedDailyDetect = []
    println("Sim #13: Low Alert Level Reff Decrease Speed: August 2021 Sim using August 2020 Fit")
    if 13 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,15)

        initCases=1

        maxCases=30*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=6
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.8]
        num_detected_before_alert=3
        alert_scaling_speed = 8

        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases =
            baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert, alert_scaling_speed)

        t_current = 6

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end
        lowAlertSpeedDailyDetect = dailyDetectCases

        # title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        # outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, Display, save, true)

        title = "August 2021 Outbreak using August 2020 Fit, Low Alert Level Reff Decrease Speed "# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryAlert_speed/EstimatedCaseNumbersAfterDetection2020FitLowAlert_speed"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, Display, save, 0.1)

        if 13 in CSVOutputRange

            timesDaily = [i for i=tspan[1]:1:tspan[end]]
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/LowAlert/BP2021fit_dailycases.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/LowAlert/BP2021fit_cumulativecases.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)
        end

    end

    println("Sim #14: Prob outbreak outside Auckland undetected")
    if 14 in simRange
        ################################################################################
        # Simulating Delta Outbreak 18 August 2021
        # time span to sim on
        tspan = (0.0,70.0)
        time_step = 1.0

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]
        numSims = convert(Int, round(10000/numSimsScaling))

        detection_tspan = (6,15) # increase from 6,14

        initCases=1

        maxCases=20*10^3
        p_test=[0.1,0.9]#p_test=[0.1,0.8]
        R_number=6
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.8] # increase from 2.2, 0.1
        num_detected_before_alert=3
        cumulativeCases, IDetect_tStep, observedIDetect, tDetect, dailyDetectCases, dailyTotalCases = baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan,
                maxCases, p_test, R_number, R_alert_scaling, t_onset_to_isol, initCases, num_detected_before_alert)

        t_current = 6

        for i in 1:length(cumulativeCases[1,:])
            for j in 1:length(cumulativeCases[:,1])
                if isnan(cumulativeCases[j,i])
                    cumulativeCases[j,i]=0.0
                end
            end
        end

        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, true, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        title = "August 2021 Outbreak using August 2020 Fit \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbersAfterDetection2020Fit"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)

        if 5 in CSVOutputRange
            @assert time_step == 1.0
            outputFileName = "August2021Outbreak/CSVOutputs/BP2021fit_dailycases.csv"
            outputCSVDailyCases(dailyDetectCases, dailyTotalCases, timesDaily, outputFileName, false)

            outputFileName = "August2021Outbreak/CSVOutputs/BP2021fit_cumulativecases.csv"
            outputCSVDailyCases(IDetect_tStep, cumulativeCases, timesDaily, outputFileName, true)
        end
    end



    # caseNumbersBeginToDrop(modPDailyDetect)

    # println("Sim #3: Heterogeneous Infection Tree Example")
    # if 3 in simRange
    #     tspan = (0.0,17.0)
    #     time_step = 0.1
    #
    #     model = init_model_pars(tspan[1], tspan[end], 5*10^6, 5*10^3, [5*10^6-1,1,0]);
    #     model.stochasticRi = true
    #     model.reproduction_number = 5
    #     model.p_test = 0.4
    #
    #     population_df = initDataframe(model);
    #     t, state_totals_all, num_cases = nextReact_branch!(population_df, model)
    #
    #     simpleGraph_branch(population_df, num_cases, true, 1)
    #
    # end
end

function main()

    # ALERT__REFF__MAX__VAL = 1.3

    compilationInit()
    # verifySolutions(1, [4])
    # verifySolutions(1, collect(5:18))
    # verifySolutions(1, collect(22:23))

    # BPbenchmarking(1, [1,2])

    # augustOutbreakSim(0.5, collect(4:11))
    # augustOutbreakSim(2, [4])
    # augustOutbreakSim(1,[5,12,13],true,false)
    # augustOutbreakSim(1,5,true,true)

    # augustOutbreakSim(1, [13.5])

    augustOutbreakPostProcess([2.72],true,true)

    # augustOutbreakSim(1, [5.5], false, false,[5.5])

    # augustOutbreakSim(1, [1], true, false,[1])

    # augustOutbreakPostProcess([2.1],true,true)


    # augustOutbreakSim(1, [5, 6, 7], true, true)
    # augustOutbreakSim(2, [5, 6,7,10,11])

    # times, dailyConfirmedCases, dailyTotalCases = reloadCSV("", true)
end

main()

if false
    # verifySolutions(1, [11,12,13,14,15])

    # model = init_model_pars(0, 200, 5*10^6, 5*10^6, [5*10^6-10,10,0])
    # time_step = 1;
    # model.p_test = 1.0
    # model.sub_clin_prop = 0
    # model.stochasticIsol = false
    # model.t_onset_shape = 5.8
    # model.t_onset_to_isol = 0

    # population_df = initDataframe(model);
    # @time t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)
    # outputFileName = "juliaGraphs/branchDiscrete/branch_model_$(model.population_size)"
    # # subtitle = "Discrete model with timestep of $time_step days"
    # # plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)
    #
    # # # next tracked heap
    # model = init_model_pars(0, 200, 5*10^6, 5*10^3, [5*10^6-10,10,0]);
    # population_df = initDataframe(model);
    # @time t, state_totals_all, num_cases = nextReact_branch_trackedHeap!(population_df, model)
    # outputFileName = "juliaGraphs/branchNextReact/branch_model_$(model.population_size)"
    # subtitle = "Next react model"
    # plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)
    #
    # # next, regular heap
    # model = init_model_pars(0, 200, 5*10^6, 5*10^6, [5*10^6-10,10,0]);
    # population_df = initDataframe(model);
    # @time t, state_totals_all, num_cases = nextReact_branch!(population_df, model)
    # outputFileName = "juliaGraphs/branchNextReact/branch_model_$(model.population_size)"
    # subtitle = "Next react model"
    # plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)

    # Simple branch
    # model = init_model_pars(0, 200, 5*10^3, 5*10^3, [5*10^3-10,10,0]);
    # population_df = initDataframe_thin(model);
    # @time t, state_totals_all, population_df = bpMain!(population_df, model, true, ThinFunction(ThinNone()), true)
    # outputFileName = "juliaGraphs/branchSimple/branch_model_$(model.population_size)"
    # subtitle = "Simple Branching Process"
    # plotSimpleBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)

    # Simple branch, random infection times
    # model = init_model_pars(0, 200, 5*10^6, 5*10^6, [5*10^6-10,10,0]);
    # population_df = initDataframe_thin(model);
    # @time t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinNone()))
    # outputFileName = "juliaGraphs/branchSimpleRandITimes/branch_model_$(model.population_size)"
    # subtitle = "Simple Branching Process, Random Infection Times"
    # plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)

    # Simple branch, random infection times, s saturation thinning
    # model = init_model_pars(0, 200, 5*10^6, 5*10^6, [5*10^6-10,10,0]);
    # population_df = initDataframe_thin(model);
    # @time t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinTree()), false)
    # outputFileName = "juliaGraphs/branchSThinRandITimes/branch_model_$(model.population_size)"
    # subtitle = "Branching Process, Random Infection Times, S saturation thinning"
    # plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)

    # Simple branch, random infection times, isolation thinning
    # model = init_model_pars(0, 200, 5*10^3, 5*10^4, [5*10^3-10,10,0]);
    # model.p_test = 1.0
    # model.sub_clin_prop = 0
    # model.stochasticIsol = false
    # # model.t_onset_shape = 5.8
    # model.t_onset_to_isol = 0
    # population_df = initDataframe_thin(model);
    # @time t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinTree()), false, true, true)
    # outputFileName = "juliaGraphs/branchSThinIsolThinRandITimes/branch_model_$(model.population_size)"
    # subtitle = "Branching Process, Random Infection Times, Isolation thinning"
    # plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)

    # Simple branch, random infection times, s saturation and isolation thinning
    # model = init_model_pars(0, 200, 5*10^5, 5*10^5, [5*10^5-10,10,0]);
    # model.p_test = 1.0;
    # model.sub_clin_prop = 0;
    # model.stochasticIsol = false;
    # # model.t_onset_shape = 5.8
    # model.t_onset_to_isol = 0;
    # population_df = initDataframe_thin(model);
    # @profiler bpMain!(population_df, model, false, ThinFunction(ThinSingle()), true, true, true);
    # outputFileName = "juliaGraphs/branchSThinIsolThinRandITimes/branch_model_$(model.population_size)"
    # subtitle = "Branching Process, Random Infection Times, S saturation & Isolation thinning"
    # plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)

    # tspan = [0,100];
    # model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
    # Simple branch, random infection times, s saturation thinning
    # population_df = initDataframe_thin(model);
    # t, state_totals_all, population_df = bpMain!(population_df, model, false, true, true, true, false)

    # time_step=0.5
    # tspan = [0,100]
    # times = [i for i=tspan[1]:time_step:tspan[end]]
    # model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-1,1,0], true);
    # model.stochasticRi = true
    # model.reproduction_number = 5
    # model.p_test = 0.4
    # model.alert_pars.R_scaling = 1.0
    #
    # population_df = initDataframe(model);
    # t, state_totals_all, num_cases = nextReact_branch!(population_df, model)
    # t_first_detect = t[findfirst(state_totals_all[:,4].==1)]
    #
    #
    # detectIndexes = findall(diff(state_totals_all[:,4]).==1) .+ 1
    # undetectedIndexes = setdiff(collect(1:length(t)), detectIndexes)
    # state_totals_all_new = state_totals_all[undetectedIndexes, 1:3]
    # tnew = t[undetectedIndexes] .- t_first_detect
    #
    # issorted(tnew)
    # issorted(state_totals_all_new)
    #
    # IDetect = state_totals_all[detectIndexes, 4]
    # tnew_detect = t[detectIndexes] .- t_first_detect
    #
    # issorted(tnew_detect)
    # issorted(IDetect)
    #
    # # interpolate using linear splines
    # StStep, ItStep, RtStep = multipleLinearSplines(state_totals_all_new, tnew, times)
    # IDetect_tStep = singleSpline(IDetect, tnew_detect, times)
    #
    #
    # issorted(t)
end


# DAY__ZERO__DATE = Date("17-08-2021", dateformat"d-m-y")
# newDate = DAY__ZERO__DATE + Dates.Day(1)
#
# Dates.format(DAY__ZERO__DATE, "u d")
# Dates.format(newDate, "u d")

# DAY__ZERO__DATE = Date("17-08-2021", dateformat"d-m-y")
# newDate = DAY__ZERO__DATE + Dates.Day(30)

# Dates.format(newDate, "u d")

# original 26 aug worst case and best case Reff in intervention
# model = init_model_pars(0,0, 0, 0, [0,0,0], true)
# model.alert_pars.p_test=0.8
# model.alert_pars.t_onset_to_isol=1.5
# model.alert_pars.R_scaling=0.25
# model.alert_pars.alert_level_scaling_speed=10
# model.reproduction_number=6.5
# println(modelReff(model))
#
# model = init_model_pars(0,0, 0, 0, [0,0,0], true)
# model.alert_pars.p_test=1.0
# model.alert_pars.t_onset_to_isol=0.5
# model.alert_pars.R_scaling=0.2
# model.alert_pars.alert_level_scaling_speed=2
# model.reproduction_number=5.5
# println(modelReff(model))
#
# ensemble = Ensemble([[0.05,0.1],[0.8,1.0]], [0.0,0.0], [[1.0,3.0],[0.5,1.5]],[0.20,0.25],[2,10],[5.5,6.5], [1], Dict())
#
# infection_time_dist = Weibull(model.t_generation_shape, model.t_generation_scale)
#
# cdf(infection_time_dist, 21)


# Current (Sep 9) worst case and best case Reff in intervention
# model = init_model_pars(0,0, 0, 0, [0,0,0], true)
# model.alert_pars.p_test=0.7
# model.alert_pars.t_onset_to_isol=1.5
# model.alert_pars.R_scaling=0.5
# model.alert_pars.alert_level_scaling_speed=10
# model.reproduction_number=6.5
# println(modelReff(model))
#
# model = init_model_pars(0,0, 0, 0, [0,0,0], true)
# model.alert_pars.p_test=1.0
# model.alert_pars.t_onset_to_isol=0.5
# model.alert_pars.R_scaling=0.01
# model.alert_pars.alert_level_scaling_speed=10
# model.reproduction_number=5.5
# println(modelReff(model))


# function reloadCSV_test(CSVpath::String, cumulative::Bool)
#
#     type = ""
#     if cumulative
#         type = "cumulative"
#     else
#         type = "daily"
#     end
#
#     # CSVpath = "August2021Outbreak/CSVOutputs/BP2021fit_$(type)cases.csv"
#
#     case_df = DataFrame(CSV.File(CSVpath))
#
#     case_df = Df_transpose(case_df)
#
#     times = case_df.time
#
#     dailyConfirmedCases = zeros(length(times), Int64((ncol(case_df)-1) /2))
#     dailyTotalCases = zeros(length(times), Int64((ncol(case_df)-1) /2))
#
#     for col in 1:length(dailyConfirmedCases[1,:])
#         colname = "$type confirmed cases - run_$col"
#         dailyConfirmedCases[:,col] = case_df[:, colname]
#
#         colname = "$type total cases - run_$col"
#         dailyTotalCases[:,col] = case_df[:, colname]
#
#     end
#
#     return times, dailyConfirmedCases, dailyTotalCases
# end


# reloadCSV("August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_11Sep_contact_transpose.csv", false, true)
# CSVpath = "August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_10Sep_contact_test.csv"
# case_df = DataFrame(CSV.File(CSVpath,transpose=false,header=2))
#
# CSVpath = "August2021Outbreak/CSVOutputs/BP2021ensemble_dailycases_10Sep_contact.csv"
# case_df = DataFrame(CSV.File(CSVpath))
#
#
# case_df = Df_transpose(case_df)
#
# # case_df.time
#
convertCSVtoConditioning("August2021Outbreak/CSVOutputs/BP2021ensemble_cumulativecases_11Sep_contact_transpose.csv", "August2021Outbreak/CSVOutputs/config_11_created11Sep.timeseries.csv", true)
