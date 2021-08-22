using BenchmarkTools
using DataFrames
using Distributions, Random, StatsBase, Statistics
using LightGraphs, GraphPlot, NetworkLayout
using PyPlot, Seaborn
using ProgressMeter

# import required modules
push!( LOAD_PATH, "./" )
using plotsPyPlot: plotBranchPyPlot, plotSimpleBranchPyPlot, plotCumulativeInfections,
    plotBenchmarksViolin
using BranchVerifySoln
using branchingProcesses

global const PROGRESS__METER__DT = 0.2

function discreteSIR_sim(time_step::Union{Float64, Int64}, numSimulations::Int64, tspan, numSimsScaling)

    # times to sim on
    times = [i for i=tspan[1]:time_step:tspan[end]]

    numSims = convert(Int, round(numSimulations / numSimsScaling))

    StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

    models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
    i = 1
    time = @elapsed Threads.@threads for i = 1:numSims

        models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

        population_df = initDataframe(models[Threads.threadid()]);
        t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

        StStep[:,i] = state_totals_all[:,1]
        ItStep[:,i] = state_totals_all[:,2]
        RtStep[:,i] = state_totals_all[:,3]

    end

    println("Finished Simulation in $time seconds")

    Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

    return hcat(Smean, Imean, Rmean), times
end


function verifySolutions(numSimsScaling::Int64, testRange)
    #=
    Discrete is the baseline. We will compare the first and next reaction outputs
    to it. With a small enough time step it should be very similar.

    1. Average reproduction number should ≈ estimated reproduction number, if
    number of cases is small relative to population size.
    - Test where all cases are clinical and have same Ri (no isolation or alert level params)
    - Test above + subclinical cases
    - Add in isolation to first test -> only clinical cases and 100% chance
    of being tested and isolated after T days (either random variable or const)
    - Test for variable Ri

    2. For each of the above cases check that the model output (epidemic curves)
    averaged over many sims are ≈ the same. Use the same techniques as we did in
    ODEVerifySoln.jl.
    =#

    println("Test #1: Reproduction Number, Deterministic Case")
    if 1 in testRange
        # Ri same for all cases and no subclin cases
        println("Deterministic Case")

        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(convert(Int, round(1000 / numSimsScaling))))
        reproduction_number = 0.0
        meanOffspring = zeros(numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^8, 10000, [5*10^8-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            models[Threads.threadid()].sub_clin_prop = 0

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)

            reproduction_number = models[Threads.threadid()].reproduction_number
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $reproduction_number")

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of First Case")
        numSims = convert(Int, round(40 / numSimsScaling))

        meanOffspring = zeros(numSims)
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^8, 3000, [5*10^8-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].recovery_time = 20

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = firstReact_branch!(population_df, models[Threads.threadid()])

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)

            reproduction_number = models[Threads.threadid()].reproduction_number
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $reproduction_number")
        println("Finished Simulation in $time seconds")
        println()
    end

    println("Test #2: Reproduction Number, Stochastic Case")
    if 2 in testRange
        # Ri same for all cases and no subclin cases
        println("Deterministic Case")

        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(1000 / numSimsScaling))
        reproduction_number = 0.0
        meanOffspring = zeros(numSims)
        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^8, 10000, [5*10^8-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            models[Threads.threadid()].sub_clin_prop = 0

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            reproduction_number = models[Threads.threadid()].reproduction_number
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $reproduction_number")

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of First Case")
        numSims = convert(Int, round(40 / numSimsScaling))

        meanOffspring = zeros(numSims)
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^8, 3000, [5*10^8-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].recovery_time = 20

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = firstReact_branch!(population_df, models[Threads.threadid()])

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            reproduction_number = models[Threads.threadid()].reproduction_number
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $reproduction_number")
        println("Finished Simulation in $time seconds")
        println()
    end

    println("Test #3: Reproduction Number, Deterministic, SubClin Case")
    if 3 in testRange
        # Ri same for all cases and no subclin cases
        println("Deterministic Case")

        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(1000 / numSimsScaling))
        reproduction_number = 0.0
        meanOffspring = zeros(numSims)
        meanRNumber = zeros(numSims)
        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^8, 10000, [5*10^8-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            models[Threads.threadid()].sub_clin_prop = 0.5

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            meanRNumber[i] = mean(inactive_df.reproduction_number)
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $(mean(meanRNumber))")
        println("Finished Simulation in $time seconds")

        println("Beginning simulation of First Case")
        numSims = convert(Int, round(40 / numSimsScaling))

        meanOffspring = zeros(numSims)
        meanRNumber = zeros(numSims)
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^8, 3000, [5*10^8-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            models[Threads.threadid()].sub_clin_prop = 0.5
            models[Threads.threadid()].recovery_time = 20

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = firstReact_branch!(population_df, models[Threads.threadid()])

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            meanRNumber[i] = mean(inactive_df.reproduction_number)

        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $(mean(meanRNumber))")
        println("Finished Simulation in $time seconds")
        println()
    end

    println("Test #4: Reproduction Number, Stochastic, SubClin Case")
    if 4 in testRange
        # Ri same for all cases and no subclin cases
        println("Deterministic Case")

        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(1000 / numSimsScaling))
        reproduction_number = 0.0
        meanOffspring = zeros(numSims)
        meanRNumber = zeros(numSims)
        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^8, 10000, [5*10^8-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            models[Threads.threadid()].sub_clin_prop = 0.5

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            meanRNumber[i] = mean(inactive_df.reproduction_number)
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $(mean(meanRNumber))")
        println("Finished Simulation in $time seconds")

        println("Beginning simulation of First Case")
        numSims = convert(Int, round(40 / numSimsScaling))

        meanOffspring = zeros(numSims)
        meanRNumber = zeros(numSims)
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^8, 3000, [5*10^8-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            models[Threads.threadid()].sub_clin_prop = 0.5
            models[Threads.threadid()].recovery_time = 20

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            meanRNumber[i] = mean(inactive_df.reproduction_number)

        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $(mean(meanRNumber))")
        println("Finished Simulation in $time seconds")
        println()
    end

    println("Test #5: Epidemic curves")
    if 5 in testRange
        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 0.02

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(200 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0])

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of First Case")
        numSims = convert(Int, round(40 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        # models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0])

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = firstReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "First React vs Discrete. Discrete timestep = $time_step"
        outputFileName = "./verifiedBranch/FirstVsDiscrete"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, true, true, true)
        println()
    end

    println("Test #6: Epidemic curves (Next vs Discrete)")
    if 6 in testRange
        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 0.02

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(200 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of Next Case")
        numSims = convert(Int, round(200 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "Next React vs Discrete. Discrete timestep = $time_step"
        outputFileName = "./verifiedBranch/NextvsDiscrete"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true)
        println()
    end

    println("Test #7: Epidemic curves (Next vs Discrete - 1 Day timestep)")
    if 7 in testRange
        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(200 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of Next Case")
        numSims = convert(Int, round(200 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "Next React vs Discrete. Discrete timestep = $time_step"
        outputFileName = "./verifiedBranch/NextvsDiscrete1Day"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true)
        println()
    end

    println("Test #8: Epidemic curves - changing Time Steps")
    if 8 in testRange
        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = [1, 0.2, 0.02]
        numSims = 200

        discrete_mean_1, times1 = discreteSIR_sim(time_step[1], numSims, tspan, numSimsScaling)
        discrete_mean_2, times2 = discreteSIR_sim(time_step[2], numSims, tspan, numSimsScaling)
        discrete_mean_3, times3 = discreteSIR_sim(time_step[3], numSims, tspan, numSimsScaling)

        title = "Discrete solution for fixed inputs when varying time step"
        outputFileName = "./verifiedBranch/DiscreteVariedTimeStep"
        branchTimeStepPlot(discrete_mean_1, discrete_mean_2, discrete_mean_3, times1, times2, times3, title, outputFileName, true, true)
        println()
    end

    println("Test #9: Epidemic curves (Next vs Discrete, Isolation)")
    if 9 in testRange
        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 0.01

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(300 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            models[Threads.threadid()].p_test = 1.0
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].stochasticIsol = false
            # models[Threads.threadid()].t_onset_shape = 5.8
            models[Threads.threadid()].t_onset_to_isol = 0

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of Next Case")
        numSims = convert(Int, round(400 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            models[Threads.threadid()].p_test = 1.0
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].stochasticIsol = false
            # models[Threads.threadid()].t_onset_shape = 5.8
            models[Threads.threadid()].t_onset_to_isol = 0

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "Next React vs Discrete with isolation. Discrete timestep = $time_step"
        outputFileName = "./verifiedBranch/NextvsDiscreteIsolationg"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true)
        println()
    end

    println("Test #10: Epidemic curves (Next vs Discrete, Isolation - 1 Day timestep)")
    if 10 in testRange
        println("Beginning simulation of Discrete Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(200 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            models[Threads.threadid()].p_test = 1.0
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].stochasticIsol = false
            # models[Threads.threadid()].t_onset_shape = 5.8
            models[Threads.threadid()].t_onset_to_isol = 0

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = discrete_branch!(population_df, models[Threads.threadid()], time_step)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of Next Case")
        numSims = convert(Int, round(200 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            models[Threads.threadid()].p_test = 1.0
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].stochasticIsol = false
            # models[Threads.threadid()].t_onset_shape = 5.8
            models[Threads.threadid()].t_onset_to_isol = 0

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "Next React vs Discrete with isolation. Discrete timestep = $time_step"
        outputFileName = "./verifiedBranch/NextvsDiscreteIsol1Day"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true)
        println()
    end

    println("Test #11: Epidemic curves (Next vs Simple BP, S Saturation Thinning)")
    if 11 in testRange
        println("Beginning simulation of Simple BP Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 0.02

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(600 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            # Simple branch, random infection times, s saturation thinning
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinTree()), true, true, false)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of Next Case")
        numSims = convert(Int, round(600 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "Next React vs Simple Branch with Saturation Thinning"
        outputFileName = "./verifiedBranch/NextvsSimpleBP"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true, false)
        println()
    end

    println("Test #12: Epidemic curves (Next vs Simple BP, S Saturation & Isolation Thinning)")
    if 12 in testRange
        println("Beginning simulation of Simple BP Case")

        # time span to sim on
        tspan = (0.0,100.0)
        time_step = 0.01

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(600 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            # Simple branch, random infection times, s saturation and isolation thinning
            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            models[Threads.threadid()].p_test = 1.0
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].stochasticIsol = false
            # models[Threads.threadid()].t_onset_shape = 5.8
            models[Threads.threadid()].t_onset_to_isol = 0

            population_df = initDataframe_thin(models[Threads.threadid()]);
            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinTree()), false, true, true)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of Next Case")
        numSims = convert(Int, round(600 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            models[Threads.threadid()].p_test = 1.0
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].stochasticIsol = false
            # models[Threads.threadid()].t_onset_shape = 5.8
            models[Threads.threadid()].t_onset_to_isol = 0

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "Next React vs Simple BP with Isolation and S Saturation"
        outputFileName = "./verifiedBranch/NextvsSimpleBPIsolation"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true, false)
        println()
    end

    println("Test #13: Initial Epidemic curves (Next vs Simple BP, S Saturation Single Thin)")
    if 13 in testRange
        println("Beginning simulation of Simple BP Case")

        # time span to sim on
        tspan = (0.0,20.0)
        time_step = 0.02

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(600 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 1*10^3, [5*10^3-10,10,0]);

            # Simple branch, random infection times, s saturation thinning
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinSingle()), true, true, false)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of Next Case")
        numSims = convert(Int, round(600 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            if !isnothing(firstDupe)
                t = vcat(t[1:firstDupe-1], t[lastDupe:end])
                state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])
            end
            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "Next React vs Simple Branch with Saturation (Single) Thinning"
        outputFileName = "./verifiedBranch/NextvsSimpleBP_SingleThin"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true, false)
        println()
    end

    println("Test #14: Initial Epidemic curves (Next vs Simple BP, S Saturation & Isolation Single Thin)")
    if 14 in testRange
        println("Beginning simulation of Simple BP Case")

        # time span to sim on
        tspan = (0.0,20.0)
        time_step = 0.02

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(600 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            # Simple branch, random infection times, s saturation and isolation thinning
            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 1*10^3, [5*10^3-10,10,0]);
            models[Threads.threadid()].p_test = 1.0
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].stochasticIsol = false
            # models[Threads.threadid()].t_onset_shape = 5.8
            models[Threads.threadid()].t_onset_to_isol = 0

            population_df = initDataframe_thin(models[Threads.threadid()]);
            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinSingle()), false, true, true)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of Next Case")
        numSims = convert(Int, round(600 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            models[Threads.threadid()].p_test = 1.0
            models[Threads.threadid()].sub_clin_prop = 0
            models[Threads.threadid()].stochasticIsol = false
            # models[Threads.threadid()].t_onset_shape = 5.8
            models[Threads.threadid()].t_onset_to_isol = 0

            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            if !isnothing(firstDupe)
                t = vcat(t[1:firstDupe-1], t[lastDupe:end])
                state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])
            end

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, discreteSIR_mean)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

        title = "Next React vs Simple BP with Isolation and S Saturation (Single) Thinning"
        outputFileName = "./verifiedBranch/NextvsSimpleBPIsolation_SingleThin"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true, false)
        println()
    end

    println("Test #15: Cumulative Infection curve (Simple BP Infections vs Geometric Series)")
    if 15 in testRange
        println("Beginning simulation of Simple BP Case")

        # time span to sim on
        tspan = (0.0,7.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(500 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        sumOffspring = 0
        total_cases = 0

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], true, ThinFunction(ThinNone()), true, false, false)

            # filtered_df = filter(row -> row.generation_number < maximum(population_df.generation_number), population_df)
            #
            # sumOffspring += sum(filtered_df.num_offspring)
            # total_cases += nrow(filtered_df)

            # interpolate using linear splines

            StStep[1:length(t), i] = state_totals_all[:,1]
            ItStep[1:length(t), i] = state_totals_all[:,2]
            RtStep[1:length(t), i] = state_totals_all[:,3]
            # StStep[:,i], ItStep[:,i], RtStep[:,i] = state_totals_all
        end

        println("Finished Simulation in $time seconds")

        # println("average R is $(sumOffspring/total_cases)")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        meanCumI = vcat(Rmean[2:end], Rmean[end]+Imean[end])

        # discreteSIR_mean = hcat(Smean, Imean, Rmean)

        println("Solving Exponential equation")
        model = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
        # model.stochasticRi = false
        # model.sub_clin_prop = 0.0
        # model.reproduction_number = 1.0


        Iovertime = model.state_totals[2] .* ((model.reproduction_number*(1-model.sub_clin_prop) +
            model.reproduction_number*model.sub_clin_scaling*model.sub_clin_prop).^(times))

        Iovertime = cumsum(Iovertime)

        St, It, Rt = initSIRArrays(tspan, time_step, numSims)

        It = convert.(Int64, It)

        It[1,:] .= model.state_totals[2]
        time = @elapsed Threads.@threads for i in 1:numSims
            # It =
            # numCases = convert.(Int64, times .* 0.0)
            # numCases[1] = 10
            for j in Int(tspan[1]+2):Int(tspan[end]+1)
                It[j, i] = sum(rand(Poisson(3), It[j-1, i]))
            end
        end
        println("Finished Simulation in $time seconds")

        Itmean= mean(It, dims = 2)

        endminus = 2
        meanCumI = meanCumI[1:end-endminus]

        Itmean = Itmean[1:end-endminus]
        Iovertime = Iovertime[1:end-endminus]
        times = times[1:end-endminus]

        misfitI = sum(abs.(meanCumI - Iovertime))/length(meanCumI)
        println("Mean Abs Error I = $misfitI")

        title = "Geometric Series vs Simple Branch Mean"
        outputFileName = "./verifiedBranch/ExponentialvsSimpleBP"
        branch2wayVerifyPlot(meanCumI, Iovertime, times, title, outputFileName, Dots(), true, true)
        # branch2wayVerifyPlot(Itmean, Iovertime, times, title, outputFileName, true, true)
        println()
    end

    println("Test #16: Cumulative Infection curve (Simple BP Infections, Heteregenous and Non), Mean + other realisations")
    if 16 in testRange
        println("Beginning simulation of Simple BP Case, Homogeneous")

        # time span to sim on
        tspan = (0.0,9.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(400 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], true, ThinFunction(ThinNone()), true, false, false)

            StStep[1:length(t), i] = state_totals_all[:,1]
            ItStep[1:length(t), i] = state_totals_all[:,2]
            RtStep[1:length(t), i] = state_totals_all[:,3]
        end

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of Simple BP Case, Heterogeneous")

        x1 = ItStep .+ RtStep

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], true, ThinFunction(ThinNone()), true, false, false)

            StStep[1:length(t), i] = state_totals_all[:,1]
            ItStep[1:length(t), i] = state_totals_all[:,2]
            RtStep[1:length(t), i] = state_totals_all[:,3]
        end

        println("Finished Simulation in $time seconds")

        x2 = ItStep .+ RtStep

        # Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        endminus = 1
        x1 = x1[1:end-endminus, :]
        x2 = x2[1:end-endminus, :]
        times = times[1:end-endminus]

        title = "Generation Based Branching Process Simulation"
        outputFileName = "./verifiedBranch/SimpleBPHeterogeneousVsNon"
        branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName, true, true, true)

        outputFileName = "./verifiedBranch/SimpleBPHeterogeneousVsNonSD"
        branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName, false, true, true)
        println()
    end

    println("Test #17: Cumulative Infection curve (Simple BP Infections vs Geometric Series, distributed times)")
    if 17 in testRange
        println("Beginning simulation of Simple BP Case")

        # time span to sim on
        tspan = (0.0,30.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(400 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        sumOffspring = 0
        total_cases = 0

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-20,20,0]);
            models[Threads.threadid()].stochasticRi = false
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinNone()), true, false, false)

            # filtered_df = filter(row -> row.generation_number < maximum(population_df.generation_number), population_df)
            #
            # sumOffspring += sum(filtered_df.num_offspring)
            # total_cases += nrow(filtered_df)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

            # StStep[1:length(t), i] = state_totals_all[:,1]
            # ItStep[1:length(t), i] = state_totals_all[:,2]
            # RtStep[1:length(t), i] = state_totals_all[:,3]
            # StStep[:,i], ItStep[:,i], RtStep[:,i] = state_totals_all
        end

        println("Finished Simulation in $time seconds")

        # println("average R is $(sumOffspring/total_cases)")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        # determine cumulative Imean
        meanCumI = Imean + Rmean
        # meanCumI = vcat(Rmean[2:end], Rmean[end]+Imean[end])

        # discreteSIR_mean = hcat(Smean, Imean, Rmean)

        println("Solving Exponential equation")
        model = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-20,20,0]);
        model.stochasticRi = false
        # model.sub_clin_prop = 0.0
        # model.reproduction_number = 1.0

        # Iovertime = model.state_totals[2] .* (((model.reproduction_number*(1-model.sub_clin_prop) +
        #     model.reproduction_number*model.sub_clin_scaling*model.sub_clin_prop)*3.3/3).^times)

        meanGenTime = mean(Weibull(model.t_generation_shape, model.t_generation_scale))
        # normTimes = (times) ./ (meanGenTime .* log(3))
        # Inormtime = model.state_totals[2] .* ((model.reproduction_number*(1-model.sub_clin_prop) +
        #     model.reproduction_number*model.sub_clin_scaling*model.sub_clin_prop).^(normTimes))

        # times*gentime

        gen = convert.(Int64, collect(0:(tspan[end]/meanGenTime+1)))

        Iovertime = model.state_totals[2] .* ((model.reproduction_number*(1-model.sub_clin_prop) +
            model.reproduction_number*model.sub_clin_scaling*model.sub_clin_prop).^(gen))

        actualGenTimes = gen .* meanGenTime
        Iovertime = cumsum(Iovertime)

        Iovertime = singleSpline(Iovertime, actualGenTimes, times)

        # St, It, Rt = initSIRArrays(tspan, time_step, numSims)

        # It = convert.(Int64, It)

        # It[1,:] .= model.state_totals[2]
        # time = @elapsed Threads.@threads for i in 1:numSims
        #     # It =
        #     # numCases = convert.(Int64, times .* 0.0)
        #     # numCases[1] = 10
        #     for j in Int(tspan[1]+2):Int(tspan[end]+1)
        #         It[j, i] = sum(rand(Poisson(3), It[j-1, i]))
        #     end
        # end
        # println("Finished Simulation in $time seconds")

        # Itmean= mean(It, dims = 2)

        # endminus = 2
        # meanCumI = meanCumI[1:end-endminus]
        # Itmean = Itmean[1:end-endminus]
        # Iovertime = Iovertime[1:end-endminus]
        # times = times[1:end-endminus]
        # normTimes = normTimes[1:end-endminus]

        misfitI = sum(abs.(meanCumI - Iovertime))/length(meanCumI)
        println("Mean Abs Error I = $misfitI")

        title = "Geometric Series vs Simple Branch Mean, Distributed Times"
        outputFileName = "./verifiedBranch/ExponentialvsSimpleBPTimes.png"
        branch2wayVerifyPlot(meanCumI, Iovertime, times, title, outputFileName, Lines(), true, true)
        # branch2wayVerifyPlot(Inormtime, Iovertime, times, title, outputFileName, Lines(), true, false)
        println()
    end

    println("Test #18: Cumulative Infection curve (Simple BP Infections, Heteregenous and Non, distributed times), Mean + other realisations")
    if 18 in testRange
        println("Beginning simulation of Simple BP Case, Homogeneous")

        # time span to sim on
        tspan = (0.0,45.0)
        time_step = 0.2

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(300 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinNone()), true, false, false)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

        end

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of Simple BP Case, Heterogeneous")

        x1 = ItStep .+ RtStep

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinNone()), true, false, false)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

        end

        println("Finished Simulation in $time seconds")

        x2 = ItStep .+ RtStep

        endminus = 1
        x1 = x1[1:end-endminus, :]
        x2 = x2[1:end-endminus, :]
        times = times[1:end-endminus]

        title = "Time Distributed Branching Process Simulation"
        outputFileName = "./verifiedBranch/SimpleBPHeterogeneousVsNonTimes"
        branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName, true, true, true)

        outputFileName = "./verifiedBranch/SimpleBPHeterogeneousVsNonTimesSD"
        branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName, false, true, true)
        println()
    end

    println("Test #19: Cumulative Infection curve (Simple BP Infections, Heteregenous and Non, Mike et al), Mean + other realisations")
    if 19 in testRange
        println("Beginning simulation of Simple BP Case, Homogeneous")

        # time span to sim on
        tspan = (0.0,80.0)
        time_step = 0.2

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(20 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^5, 5*10^5, [5*10^5-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinTree()), true, true, false)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

        end

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of Simple BP Case, Heterogeneous")

        x1 = ItStep .+ RtStep

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^5, 5*10^5, [5*10^5-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(models[Threads.threadid()]);

            t, state_totals_all, population_df = bpMain!(population_df, models[Threads.threadid()], false, ThinFunction(ThinTree()), true, true, false)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

        end

        println("Finished Simulation in $time seconds")

        x2 = ItStep .+ RtStep

        endminus = 1
        x1 = x1[1:end-endminus, :]
        x2 = x2[1:end-endminus, :]
        times = times[1:end-endminus]

        title = "Time Distributed, Saturation Thinned Branching Process Simulation"
        outputFileName = "./verifiedBranch/SimpleBPHeterogeneousVsNonTimesSThin"
        branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName, true, true, true, 3.0)

        outputFileName = "./verifiedBranch/SimpleBPHeterogeneousVsNonTimesSThinSD"
        branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName, false, true, true, 3.0)
        println()
    end

    println("Test #20: Cumulative Infection curve (Next React BP Infections, Heteregenous and Non, Mike et al), Mean + other realisations")
    if 20 in testRange
        println("Beginning simulation of Next React BP Case, Homogeneous")

        # time span to sim on
        tspan = (0.0,80.0)
        time_step = 0.2

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(40 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^5, 5*10^5, [5*10^5-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

        end

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of Next React BP Case, Heterogeneous")

        x1 = ItStep .+ RtStep

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^5, 5*10^5, [5*10^5-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

        end

        println("Finished Simulation in $time seconds")

        x2 = ItStep .+ RtStep

        endminus = 1
        x1 = x1[1:end-endminus, :]
        x2 = x2[1:end-endminus, :]
        times = times[1:end-endminus]

        title = "Next Reaction Branching Process Simulation"
        outputFileName = "./verifiedBranch/NextHeterogeneousVsNon"
        branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName, true, true, true, 2.0)

        outputFileName = "./verifiedBranch/NextHeterogeneousVsNonSD"
        branchSideBySideVerifyPlot(x1, x2, times, title, outputFileName, false, true, true, 2.0)
        println()
    end

    println("Test #21: Cumulative Infection curve (Next React BP Infections, Heteregenous and Non, Mike et al, Isolation and Non), Mean + other realisations")
    if 21 in testRange

        println("Beginning simulation of Next React BP Case, Homogeneous")

        # time span to sim on
        tspan = (0.0,80.0)
        time_step = 0.2

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(400 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        p = Progress(numSims,PROGRESS__METER__DT)
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
            next!(p)
        end

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of Next React BP Case, Heterogeneous")

        x11 = ItStep .+ RtStep

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        p = Progress(numSims,PROGRESS__METER__DT)
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
            next!(p)
        end

        println("Finished Simulation in $time seconds")

        x21 = ItStep .+ RtStep

        println("Beginning simulation of Next React BP Case, Homogeneous Isolation")

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        p = Progress(numSims,PROGRESS__METER__DT)
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = false
            models[Threads.threadid()].p_test = 1.0
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

            next!(p)
        end

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of Next React BP Case, Heterogeneous Isolation")

        x12 = ItStep .+ RtStep

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        p = Progress(numSims,PROGRESS__METER__DT)
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            models[Threads.threadid()].stochasticRi = true
            models[Threads.threadid()].p_test = 1.0
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
            next!(p)
        end

        println("Finished Simulation in $time seconds")

        x22 = ItStep .+ RtStep

        endminus = 1
        x11 = x11[1:end-endminus, :]
        x12 = x12[1:end-endminus, :]
        x21 = x21[1:end-endminus, :]
        x22 = x22[1:end-endminus, :]
        times = times[1:end-endminus]

        title = "Next Reaction Branching Process Simulation With Isolation"
        outputFileName = "./verifiedBranch/NextHeterogeneousVsNonIsol"
        branchSideBySideVerifyPlot2(x11, x12, x21, x22, times, title, outputFileName, true, true, true, 2.0)

        outputFileName = "./verifiedBranch/NextHeterogeneousVsNonSDIsol"
        branchSideBySideVerifyPlot2(x11, x12, x21, x22, times, title, outputFileName, false, true, true, 2.0)
        println()
    end
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

function quantile2D(x, quantileValue)
    # x is 2D

    quantiles = zeros(length(x[:,1]))

    for i in 1:length(x[:,1])

        quantiles[i] = quantile(x[i,:], quantileValue)
    end
    return quantiles
end

function plotDailyCasesOutbreak(dailyConfirmedCases, timesDaily, actualDailyCumCases, timesActual, title, outputFileName, two21::Bool=true, Display=true, save=false)

    actualDailyCases = diff(vcat([0],actualDailyCumCases))
    # timesActual = timesActual[1:end-1]
    # actualDailyCases = actualDailyCumCases
    # timesActual = timesActual

    # dailyIndexes = findall(rem.(times, 1).==0)
    # dailyIndexes = collect(1:length(times))

    # dailyConfirmedCases = diff(confirmedCases[dailyIndexes, :],dims=1)
    # dailyConfirmedCases = confirmedCases[dailyIndexes, :]

    # medDailyConfirmedCases = diff(median(confirmedCases,dims=2)[dailyIndexes, :],dims=1)
    # dailyTimes = times[dailyIndexes][1:end-1]
    # dailyTimes = times[dailyIndexes]
    dailyTimes=timesDaily

    # # add on zeroness
    # dailyTimes = vcat(collect(-6:-1), dailyTimes)
    # dailyConfirmedCases = vcat(convert.(Int64, zeros(length(dailyIndexes),6)), dailyConfirmedCases)

    if two21 && false
        finalIndex = findfirst(dailyTimes.==15)

        dailyConfirmedCases = dailyConfirmedCases[1:finalIndex, :]
        dailyTimes = dailyTimes[1:finalIndex]
    end

    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(dpi=300)

    quantiles1 = quantile2D(dailyConfirmedCases, 0.025)
    quantiles3 = quantile2D(dailyConfirmedCases, 0.975)
    plt.plot(dailyTimes, median(dailyConfirmedCases,dims=2), "b-", label="Median Daily Confirmed Cases", lw=2.5, figure=fig)
    plt.plot(timesActual, actualDailyCases, "k-", label="Actual Daily Confirmed Cases", lw=2.5, figure=fig)


    # plt.plot(dailyTimes, quantiles1, "k--", label="95% Quantile Bands ", lw=2, alpha = 0.5)
    # plt.plot(dailyTimes, quantiles3, "k--", lw=2, alpha = 0.5)
    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.3, color = "r")

    # plt.plot(times, x2, "r$x2PlotType", label="I - Geometric Series", lw=1.5, figure=fig, alpha = 1)

    if two21
        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
        122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
        184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]

        observedIDetect = diff(vcat([0],observedIDetect))
        tDetect = collect(0:length(observedIDetect)-1)

        plt.plot(tDetect, observedIDetect, color="tab:gray", linestyle="-", label="August 2020 Daily Confirmed Cases", lw=2.5, figure=fig)
    end

    if two21 && false
        plt.xticks([-3,0,3,6,9,12,15],vcat(["Aug $(14+3*i)" for i in 0:5], ["Sep 01"]))
        plt.xlabel("Date")
    else
        plt.xlabel("Days since Detection")
    end
    plt.ylabel("Daily Confirmed Cases")
    # plt.suptitle("Branching Process Simulation")
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

function plotDailyCasesMultOutbreak(dailyConfirmedCasesHigh, dailyConfirmedCasesMod, dailyConfirmedCasesLow, timesDaily, actualDailyCumCases, timesActual, title, outputFileName, two21::Bool=true, Display=true, save=false)

    actualDailyCases = diff(vcat([0],actualDailyCumCases))

    dailyTimes=timesDaily


    Seaborn.set()
    set_style("ticks")
    Seaborn.set_color_codes("pastel")
    fig = plt.figure(dpi=300)

    # High ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesHigh, 0.025)
    quantiles3 = quantile2D(dailyConfirmedCasesHigh, 0.975)
    plt.plot(dailyTimes, median(dailyConfirmedCasesHigh,dims=2), "b-", label="Median Daily Confirmed Cases - High", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.3, color = "b")
    ######################################################################

    # Mod ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesMod, 0.025)
    quantiles3 = quantile2D(dailyConfirmedCasesMod, 0.975)
    plt.plot(dailyTimes, median(dailyConfirmedCasesMod,dims=2), "r-", label="Median Daily Confirmed Cases - Moderate", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.3, color = "r")
    ######################################################################

    # Low ######################################################################
    quantiles1 = quantile2D(dailyConfirmedCasesLow, 0.025)
    quantiles3 = quantile2D(dailyConfirmedCasesLow, 0.975)
    plt.plot(dailyTimes, median(dailyConfirmedCasesLow,dims=2), "g-", label="Median Daily Confirmed Cases - Low", lw=2.5, figure=fig)

    plt.fill_between(dailyTimes, quantiles1, quantiles3, alpha=0.3, color = "g")
    ######################################################################

    plt.plot(timesActual, actualDailyCases, "k-", label="Actual Daily Confirmed Cases", lw=2.5, figure=fig)

    # plt.plot(times, x2, "r$x2PlotType", label="I - Geometric Series", lw=1.5, figure=fig, alpha = 1)

    if two21
        observedIDetect = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
        122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
        184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]

        observedIDetect = diff(vcat([0],observedIDetect))
        tDetect = collect(0:length(observedIDetect)-1)

        plt.plot(tDetect, observedIDetect, color="tab:gray", linestyle="-", label="August 2020 Daily Confirmed Cases", lw=2.5, figure=fig)
    end

    if two21 && false
        plt.xticks([-3,0,3,6,9,12,15],vcat(["Aug $(14+3*i)" for i in 0:5], ["Sep 01"]))
        plt.xlabel("Date")
    else
        plt.xlabel("Days since Detection")
    end
    plt.ylabel("Daily Confirmed Cases")
    # plt.suptitle("Branching Process Simulation")
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


function plotAndStatsOutbreak(confirmedCases, cumulativeCases, times, observedIDetect,
    tDetect, t_current, title, outputFileName, Display=true, save=false, alphaMultiplier=1.0)
    #=
    Plot multiple realisations of confirmedCases and x2 as well as their means.
    =#

    Seaborn.set()
    set_style("ticks")
    set_color_codes("pastel")
    # fig = plt.figure(dpi=300)

    # Initialise plots - need figure size to make them square and nice
    f,ax = Seaborn.subplots(1,2, figsize=(10,6), dpi=300)

    timesVector = []
    for i in 1:length(confirmedCases[1,:])
        timesVector = vcat(timesVector, times)
    end

    # Confirmed cases since first detect #######################################
    for i in 1:length(confirmedCases[1,:])
        if i == 1
            ax[1].plot(times, confirmedCases[:,i], "b-", label="BPM Realisations", lw=2, alpha = 0.2)
        else
            ax[1].plot(times, confirmedCases[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
        end
    end
    labelconfirmedCases = ["Mean and SD" for i in 1:length(timesVector)]

    quantiles1 = quantile2D(confirmedCases, 0.025)
    quantiles3 = quantile2D(confirmedCases, 0.975)

    ax[1].plot(times, median(confirmedCases, dims=2), "r-", label="Median Realisation", lw=3, alpha = 1)
    ax[1].plot(times, quantiles1, "k--", label="95% Quantile Bands ", lw=2, alpha = 0.5)
    ax[1].plot(times, quantiles3, "k--", lw=2, alpha = 0.5)
    ax[1].fill_between(times, quantiles1, quantiles3, alpha=0.3, color = "r")
    # Seaborn.lineplot(x = timesVector, y = [confirmedCases...], hue = labelconfirmedCases, palette = "flare", ci="sd", ax=ax[1])

    ax[1].plot(tDetect, observedIDetect, "k-", label="August Outbreak", lw=2, alpha = 1)
    ############################################################################

    # 2020 Confirmed Cases
    if outputFileName == "./August2021Outbreak/EstimatedCaseNumbersAfterDetection2020Fit"
        observedIDetect2020 = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,
            122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,
            184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193,193,193,193,193,193,193]
        tDetect2020 = collect(0:length(observedIDetect2020)-1)

        ax[1].plot(tDetect2020, observedIDetect2020, color="b", linestyle="-.", label="August Outbreak 2020", lw=2, alpha = 1)
    end

    # Cumulative cases since first detect ######################################
    for i in 1:length(cumulativeCases[1,:])
        if i == 1
            ax[2].plot(times, cumulativeCases[:,i], "b-", label="BPM Realisations", lw=2, alpha = 0.2)
        else
            ax[2].plot(times, cumulativeCases[:,i], "b-", lw=2, alpha = 0.09*alphaMultiplier)
        end
    end
    labelcumulativeCases = ["Mean and SD" for i in 1:length(timesVector)]
    ax[2].plot(times, median(cumulativeCases, dims=2), "r-", label="Median Realisation", lw=3, alpha = 1)

    quantiles1 = quantile2D(cumulativeCases, 0.025)
    quantiles3 = quantile2D(cumulativeCases, 0.975)

    # ax[1].plot(times, median(confirmedCases, dims=2), "b-", label="Median Realisation", lw=4, alpha = 1)
    ax[2].plot(times, quantiles1, "k--", label="95% Quantile Bands ", lw=2, alpha = 0.5)
    ax[2].plot(times, quantiles3, "k--", lw=2, alpha = 0.5)
    ax[2].fill_between(times, quantiles1, quantiles3, alpha=0.3, color = "r")

    # Seaborn.lineplot(x = timesVector, y = [cumulativeCases...], hue = labelcumulativeCases, palette = "flare", ci="sd", ax=ax[2], estimator=median)
    ############################################################################

    if t_current > 0
        ax[1].plot([t_current, t_current], [minimum(observedIDetect), maximum(cumulativeCases)],
            "k--", label="21/8/2021", lw=2, alpha = 1)
        ax[2].plot([t_current, t_current], [minimum(observedIDetect), maximum(cumulativeCases)],
            "k--", label="21/8/2021", lw=2, alpha = 1)
    end
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
    ax[1].legend(loc = "lower right")
    ax[1].set_xlabel("Time from Detection (days)")
    ax[1].set_ylabel("Cumulative Confirmed Cases")
    ax[1].set_title("Confirmed Cases")
    ax[1].set(yscale="log")
    ax[1].set_yticks([1.0,10.0,100.0,1000.0, 10000.0])
    ax[1].set_yticklabels([1,10,100,1000,10000])

    # ax[2].set_ylim([0, maxy*0.6])
    ax[2].legend(loc = "lower right")
    ax[2].set_xlabel("Time from Detection (days)")
    ax[2].set_ylabel("Cumulative Total Cases")
    ax[2].set_title("Total Cases")
    ax[2].set(yscale="log")
    ax[2].set_yticks([1.0,10.0,100.0,1000.0, 10000.0])
    ax[2].set_yticklabels([1,10,100, 1000,10000])


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

function baseOutbreakSim(tspan, time_step, times, numSims, detection_tspan, maxCases=20*10^3, p_test::Array{Float64,1}=[0.1,0.8], R_number=3, R_alert_scaling=0.2,
    t_onset_to_isol::Union{Array{Int64,1},Array{Float64,1}}=[2.2,1], initCases=1, num_detected_before_alert=1)
    StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)
    IDetect_tStep = StStep .* 0

    dailyDetectCases = initSIRArrays(tspan, 1, numSims)[1]
    timesDaily = [i for i=tspan[1]:1:tspan[end]]

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

        # models[Threads.threadid()].sub_clin_prop = 0.0
        # models[Threads.threadid()].reproduction_number = 1.0

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

                # zeroindex = findfirst(abs.(abs.(tnew_detect).-tspan[1]) .< 1*10-8)# = tspan[1]*1
                # tnew_detect[zeroindex] = tspan[1]*1

                # interpolate using linear splines
                # StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all_new, tnew, times)
                StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, tnew, times)


                if length(IDetect) > 1
                    try
                        IDetect_tStep[:,i] = singleLinearSpline(IDetect, tnew_detect, times)
                        dailyDetectCases[:,i] = singleLinearSpline(IDetect, tnew_detect, timesDaily)
                    catch
                        println("IDetect is sorted = $(issorted(IDetect))")
                        println("Tnew is sorted = $(issorted(tnew_detect))")
                        println(tnew_detect)
                        @warn "Not good"
                    end
                else
                    # index = findfirst(t_first_detect.<times) + 1

                    IDetect_tStep[:, i] .= 1
                end
            else
                establishedSims[i] = false
            end
        else
            establishedSims[i] = false
        end

        next!(p)
    end

    # dailyDetectCases = diff(hcat(convert.(Int64, zeros(length(timesDaily))), dailyDetectCases),dims=1)
    dailyDetectCases = diff(vcat(transpose(zeros(numSims)), dailyDetectCases),dims=1)



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
    observedIDetect = cumsum([5,4,15,20,26])
    tDetect = collect(0:length(observedIDetect)-1)
    return IcumCases[:,indexesToKeep], IDetect_tStep[:,indexesToKeep], observedIDetect, tDetect, dailyDetectCases[:,indexesToKeep]
end

function augustOutbreakSim(numSimsScaling::Union{Float64,Int64}, simRange)
    #=
    Estimation of the August 2021 Delta outbreak on 18 Aug.

    1 seed case, estimated to have been seeded 10 to 17 days ago.

    Estimate the cumulative number of infections from this time point.
    =#

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
        plotCumulativeInfections(x[:,indexesToKeep], times, [10.0,13.0,15.0,17.0], title, outputFileName, true, true, 0.1)
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
        plotCumulativeInfections(x[:,indexesToKeep], times, [10.0,13.0,15.0,17.0], title, outputFileName, true, true, 0.1)


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
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.1]
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

    modPDailyDetect = []
    println("Sim #5: August 2021 Sim using August 2020 Fit")
    if 5 in simRange
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

        modPDailyDetect = dailyDetectCases

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "August 2021 Outbreak using August 2020 Fit, Daily Case Numbers After Detection"
        outputFileName = "./August2021Outbreak/DailyCaseNumbersAfterDetection2020Fit"
        plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, true, true)
        # plotDailyCasesOutbreak(dailyDetectCases, timesDaily, observedIDetect, tDetect, title, outputFileName, true, false, true)

        # title = "August 2021 Outbreak using August 2020 Fit \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        # outputFileName = "./August2021Outbreak/EstimatedCaseNumbersAfterDetection2020Fit"
        # plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
    end

    println("Sim #5.5: August 2021 Sim using August 2020 Fit")
    if 5.5 in simRange
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
        R_number=6
        # R_alert_scaling=0.2
        # t_onset_to_isol=[2.2,1.0]
        R_alert_scaling=0.25
        t_onset_to_isol=[2.2,0.1]
        num_detected_before_alert=2
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
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, true, true, true)

        title = "August 2021 Outbreak using August 2020 Fit, Low R0"# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryR_i/EstimatedCaseNumbersAfterDetection2020FitLowR"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
    end

    println("Sim #7.5: Effect of R Pre scaling on Daily Cases: August 2021 Sim using August 2020 Fit")
    if 5 in simRange && 6 in simRange && 6 in simRange

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Varying Pre Intervention Reff"
        outputFileName = "./August2021Outbreak/VaryR_i/DailyCaseNumbersAfterDetection"
        plotDailyCasesMultOutbreak(highRPreDailyDetect, modPDailyDetect, lowRPreDailyDetect, timesDaily, observedIDetect, tDetect, title, outputFileName, true, true, true)
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
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, true, true, true)

        title = "August 2021 Outbreak using August 2020 Fit, High P_test"# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryP_test/EstimatedCaseNumbersAfterDetection2020FitHighP_test"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
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
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, true, true, true)

        title = "August 2021 Outbreak using August 2020 Fit, Low P_test"# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryP_test/EstimatedCaseNumbersAfterDetection2020FitLowP_test"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
    end

    println("Sim #9.5: Effect of P_test scaling on Daily Cases: August 2021 Sim using August 2020 Fit")
    if 5 in simRange && 8 in simRange && 9 in simRange

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Varying Symptomatic Testing Rate"
        outputFileName = "./August2021Outbreak/VaryP_test/DailyCaseNumbersAfterDetection"
        plotDailyCasesMultOutbreak(highPDailyDetect, modPDailyDetect, lowPDailyDetect, timesDaily, observedIDetect, tDetect, title, outputFileName, true, true, true)
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
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, true, true, true)

        title = "August 2021 Outbreak using August 2020 Fit, High Alert Level Reduction of R0 "# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryR_scaling/EstimatedCaseNumbersAfterDetection2020FitHighAlert_Reduct"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
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
        # plotDailyCasesOutbreak(IDetect_tStep, times, observedIDetect, tDetect, title, outputFileName, true, true, true)

        title = "August 2021 Outbreak using August 2020 Fit, Low Alert Level Reduction of R0 "# \n Estimated Case Numbers After Detection of 3 Cases on Day Zero (Actual is 5)"
        outputFileName = "./August2021Outbreak/VaryR_scaling/EstimatedCaseNumbersAfterDetection2020FitLowAlert_Reduct"
        plotAndStatsOutbreak(IDetect_tStep, cumulativeCases, times, observedIDetect, tDetect, t_current, title, outputFileName, true, true, 0.1)
    end

    println("Sim #11.5: Effect of R Post scaling on Daily Cases: August 2021 Sim using August 2020 Fit")
    if 5 in simRange && 10 in simRange && 11 in simRange

        timesDaily = [i for i=tspan[1]:1:tspan[end]]
        title = "Daily Case Numbers After Detection, Varying Post Intervention Reff"
        outputFileName = "./August2021Outbreak/VaryR_scaling/DailyCaseNumbersAfterDetection"
        plotDailyCasesMultOutbreak(highRPostDailyDetect, modPDailyDetect, lowRPostDailyDetect, timesDaily, observedIDetect, tDetect, title, outputFileName, true, true, true)
    end


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



# function R_estimateAtTimes(args)
#     body
# end


function main()

    compilationInit()
    # verifySolutions(1, 4)
    # verifySolutions(1, collect(5:18))
    # verifySolutions(1, collect(21))

    # BPbenchmarking(1, [1,2])

    # augustOutbreakSim(0.5, collect(4:11))

    augustOutbreakSim(1, [1,2])
    # augustOutbreakSim(2, [5, 6,7,10,11])
end

main()

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
