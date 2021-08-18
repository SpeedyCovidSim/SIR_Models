using BenchmarkTools
using DataFrames
using Distributions, Random, StatsBase
using LightGraphs, GraphPlot, NetworkLayout
using PyPlot, Seaborn

# import required modules
push!( LOAD_PATH, "./" )
using plotsPyPlot: plotBranchPyPlot, plotSimpleBranchPyPlot, plotCumulativeInfections
using BranchVerifySoln
using branchingProcesses

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
            t, state_totals_all, num_cases = firstReact_branch!(population_df, models[Threads.threadid()])

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

    println("Test #15: Epidemic curves (Simple BP Infections vs Exponential)")
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

    println("Test #16: Epidemic curves (Simple BP Infections, Heteregenous and Non), Mean + other realisations")
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

    println("Test #17: Epidemic curves (Simple BP Infections vs Exponential, distributed times)")
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

    println("Test #18: Epidemic curves (Simple BP Infections, Heteregenous and Non, distributed times), Mean + other realisations")
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

    println("Test #19: Epidemic curves (Simple BP Infections, Heteregenous and Non, Mike et al), Mean + other realisations")
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

    println("Test #20: Epidemic curves (Next React BP Infections, Heteregenous and Non, Mike et al), Mean + other realisations")
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

    println("Test #21: Epidemic curves (Next React BP Infections, Heteregenous and Non, Mike et al, Isolation and Non), Mean + other realisations")
    if 21 in testRange

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

        x11 = ItStep .+ RtStep

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

        x21 = ItStep .+ RtStep

        println("Beginning simulation of Next React BP Case, Homogeneous Isolation")

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^5, 5*10^5, [5*10^5-10,10,0]);
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

        end

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of Next React BP Case, Heterogeneous Isolation")

        x12 = ItStep .+ RtStep

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^5, 5*10^5, [5*10^5-10,10,0]);
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

function augustOutbreakSim(numSimsScaling::Int64, simRange)
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

        numSims = convert(Int, round(5000/numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^6, 5*10^3, [5*10^6-1,1,0]);
            models[Threads.threadid()].stochasticRi = true
            models[Threads.threadid()].reproduction_number = 5
            models[Threads.threadid()].p_test = 0.4
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            # firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            # lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            # t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            # state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

        end

        println("Finished Simulation in $time seconds")
        x = ItStep .+ RtStep

        title = "August 2021 Outbreak Cumulative Infection Estimation"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbers"
        plotCumulativeInfections(x, times, [10.0,13.0,15.0,17.0], title, outputFileName, true, true, 0.1)
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

        numSims = convert(Int, round(5000/numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        models = [init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]) for i in 1:Threads.nthreads()]
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            models[Threads.threadid()] = init_model_pars(tspan[1], tspan[end], 5*10^6, 5*10^3, [5*10^6-1,1,0]);
            models[Threads.threadid()].stochasticRi = false
            models[Threads.threadid()].reproduction_number = 5
            models[Threads.threadid()].p_test = 0.4
            # models[Threads.threadid()].sub_clin_prop = 0.0
            # models[Threads.threadid()].reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe(models[Threads.threadid()]);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, models[Threads.threadid()])

            # clean duplicate values of t which occur on the first recovery time
            # firstDupe = findfirst(x->x==models[Threads.threadid()].recovery_time,t)
            # lastDupe = findlast(x->x==models[Threads.threadid()].recovery_time,t)

            # t = vcat(t[1:firstDupe-1], t[lastDupe:end])
            # state_totals_all = vcat(state_totals_all[1:firstDupe-1, :], state_totals_all[lastDupe:end, :])

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)

        end

        println("Finished Simulation in $time seconds")
        x = ItStep .+ RtStep

        title = "August 2021 Outbreak Cumulative Infection Estimation"
        outputFileName = "./August2021Outbreak/EstimatedCaseNumbersHomogeneous"
        plotCumulativeInfections(x, times, [10.0,13.0,15.0,17.0], title, outputFileName, true, true, 0.1)


        ################################################################################
    end
end

compilationInit()
# verifySolutions(1, 5)
# verifySolutions(1, collect(5:18))
# verifySolutions(1, collect(21))

augustOutbreakSim(1, collect(1:2))

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


# tspan = [0,100]
# model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
#
# population_df = initDataframe(model);
# t, state_totals_all, num_cases = firstReact_branch!(population_df, model)
#
# model.state_totals
