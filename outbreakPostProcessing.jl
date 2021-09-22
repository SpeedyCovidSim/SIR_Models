module outbreakPostProcessing

    # using BenchmarkTools
    using DataFrames
    using Distributions, Random, StatsBase, Statistics
    # using LightGraphs, GraphPlot, NetworkLayout
    using PyPlot, Seaborn
    # using ProgressMeter
    using CSV
    using Dates
    # using PlotlyJS
    using branchingProcesses: branchModel

    export createNewDir, quantile2D, probOfLessThanXGivenYDays, outputCSVDailyCases,
        outputCSVmodels, reloadCSVmodels, reloadCSV, removeNaNs, modelReff, Df_transpose, convertCSVtoConditioning

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

    function modelReff(model::branchModel)::AbstractFloat
        #=
        Given a set of branching process model parameters, return the equivalent Reff
        value that occurs when within an intervention state.
        =#

        infection_dist = Weibull(model.t_generation_shape, model.t_generation_scale)

        T1 = mean(Gamma(model.t_onset_shape, model.t_onset_scale))
        T2 = model.alert_pars.t_onset_to_isol * 1

        q = cdf(infection_dist, T1 + T2)

        return model.alert_pars.R_scaling * model.reproduction_number * (
            (1-model.sub_clin_prop) * (1-model.alert_pars.p_test +
                model.alert_pars.p_test*(q + model.isolation_infectivity*(1-q))) +
            model.sub_clin_prop * (1-model.alert_pars.p_test_sub +
                model.alert_pars.p_test_sub*(q + model.isolation_infectivity*(1-q))) * model.sub_clin_scaling)

    end

    function createNewDir(path, daily=false)
        #=
        Given a path, determine if a folder has been created for the current date in
        that path. If not, create one - this will allow seperation of figures created
        on different days.
        =#

        currentTime = now()
        folderName = Dates.format(currentTime, "Y_m_d_u")

        newdir = path*"/"*folderName

        if daily
            newdir = newdir*"_daily"
        end

        if !isdir(newdir)
            mkdir(newdir)
        end

        return newdir
    end

    function quantile2D(x, quantileValue)
        # x is 2D

        quantiles = zeros(length(x[:,1]))

        for i in 1:length(x[:,1])

            quantiles[i] = quantile(x[i,:], quantileValue)
        end
        return quantiles
    end

    function indexWhereAllValuesLessThanX(array, x)
        #=
        Given an array, return the largest range that represents the range array[index:end]
        where all values in this range are less than x

        If no range exists, return 0:0
        =#
        for i in length(array):-1:1
            if array[i] >= x
                if i+1 > length(array)
                    return 0:0
                else
                    return i+1:length(array)
                end
            end
        end

        return 1:length(array)
    end

    function testIndexWhereAll()
        @assert indexWhereAllValuesLessThanX([1,2,3,4,5], 6) == 1:5
        @assert indexWhereAllValuesLessThanX([5,4,3,2,1], 4) == 3:5
    end

    function probOfLessThanXGivenYDays(dailyConfirmedCases, x, y::Union{UnitRange,StepRange})
        #=
        Given a 2D array, where the columns contain individual realisations and the
        rows represent values at given times, return the proportion of columns that
        satisfy indexWhereAllValuesLessThanX after y days.

        The first row corresponds to t=0
        The second row corresponds to t=1 etc.
        =#

        probability = zeros(length(y))

        caseXranges = [0:0 for _ in 1:length(dailyConfirmedCases[1,:])]

        for col in 1:length(dailyConfirmedCases[1,:])
            caseXranges[col] = indexWhereAllValuesLessThanX(dailyConfirmedCases[:,col], x)
        end

        for i in 1:length(y)
            for j in caseXranges
                if !isnothing(j) && y[i] in j
                    probability[i] += 1.0
                end
            end
            probability[i] = probability[i] / length(caseXranges)
        end

        return probability
    end

    function testprobOfLess()
        @assert sum(probOfLessThanXGivenYDays([0 1; 2 3; 5 6; 4 5; 3 2; 2 1; 1 0], 6, 1:7) .== [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])==7
        @assert sum(probOfLessThanXGivenYDays([0 1; 2 3; 5 6; 4 5; 3 2; 2 1; 1 0], 5, 1:7) .== [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])==7
    end

    function Df_transpose(df)
        #=
        Given a DataFrame, return it's transpose
        =#
        return DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])
    end

    function outputCSVDailyCases(dailyConfirmedCases, dailyTotalCases, times, outputFileName, cumulative::Bool, transpose_df::Bool=false)

        case_df = DataFrame()
        case_df[!, "time"] = times
        case_df.time = convert.(Int64, case_df.time)

        type = ""
        if cumulative
            type = "cumulative"
        else
            type = "daily"
        end

        for col in 1:length(dailyConfirmedCases[1,:])

            colname = "$type confirmed cases - run_$col"
            case_df[!, colname] = dailyConfirmedCases[:,col]

            colname = "$type total cases - run_$col"
            case_df[!, colname] = dailyTotalCases[:,col]

        end

        # outputFileName = "August2021Outbreak/CSVOutputs/BP2021fit_$(type)cases.csv"

        if transpose_df
            case_df = Df_transpose(case_df)
        end

        CSV.write(outputFileName, case_df)

        return nothing
    end

    function outputCSVmodels(models, outputFileName, meanOffspring=[], alertIncrease=[], usingContactTrace=false)

        infection_dist = Weibull(models[1].t_generation_shape, models[1].t_generation_scale)

        models_df = DataFrame()

        models_df.reproduction_number = zeros(length(models))
        models_df.r_scaling = zeros(length(models))
        models_df.sub_clin_prop = zeros(length(models))
        models_df.sub_clin_scaling = zeros(length(models))
        models_df.t2_before_al = zeros(length(models))
        models_df.t2_after_al = zeros(length(models))
        models_df.p_test_before_al = zeros(length(models))
        models_df.p_test_after_al = zeros(length(models))
        models_df.p_test_sub_after_al = zeros(length(models))

        models_df.R_eff_after_al = zeros(length(models))


        if !isempty(alertIncrease)
            models_df.alertIncrease = alertIncrease
        end

        for row in 1:length(models)

            models_df[row, :reproduction_number] = models[row].reproduction_number
            models_df[row, :r_scaling] = models[row].alert_pars.R_scaling
            models_df[row, :sub_clin_prop] = models[row].sub_clin_prop
            models_df[row, :sub_clin_scaling] = models[row].sub_clin_scaling
            models_df[row, :t2_before_al] = models[row].t_onset_to_isol
            models_df[row, :t2_after_al] = models[row].alert_pars.t_onset_to_isol
            models_df[row, :p_test_before_al] = models[row].p_test
            models_df[row, :p_test_after_al] = models[row].alert_pars.p_test
            models_df[row, :p_test_sub_after_al] = models[row].alert_pars.p_test_sub

            T1 = mean(Gamma(models[row].t_onset_shape, models[row].t_onset_scale))
            T2 = models[row].alert_pars.t_onset_to_isol * 1

            q = cdf(infection_dist, T1 + T2)

            models_df[row, :R_eff_after_al] = models_df[row, :r_scaling] * models_df[row,:reproduction_number] * (
                (1-models_df[row, :sub_clin_prop]) * (1-models_df[row, :p_test_after_al] +
                    models_df[row, :p_test_after_al]*(q + models[row].isolation_infectivity*(1-q))) +
                models_df[row, :sub_clin_prop] * (1-models_df[row, :p_test_sub_after_al] +
                    models_df[row, :p_test_sub_after_al]*(q + models[row].isolation_infectivity*(1-q))) * models_df[row, :sub_clin_scaling])

        end

        if !isempty(meanOffspring)
            models_df.R_eff_after_al_estimate = meanOffspring
        end

        models_df.R_eff_before_al = models_df.reproduction_number .* ((1.0 .- models_df.sub_clin_prop) .+ models_df.sub_clin_scaling .* models_df.sub_clin_prop)

        models_df.num_seed_cases = [models[i].state_totals[2]*1 for i in 1:length(models)]
        models_df.alert_scaling_speed = [models[i].alert_pars.alert_level_scaling_speed*1 for i in 1:length(models)]

        if usingContactTrace
            models_df.p_contact = [models[i].alert_pars.p_contact*1 for i in 1:length(models)]
            models_df.p_contact_time = [models[i].alert_pars.p_contact_time*1 for i in 1:length(models)]
        end

        CSV.write(outputFileName, models_df)

        return nothing
    end

    function reloadCSV(CSVpath::String, cumulative::Bool, transposed_df::Bool=false)

        type = ""
        if cumulative
            type = "cumulative"
        else
            type = "daily"
        end

        # CSVpath = "August2021Outbreak/CSVOutputs/BP2021fit_$(type)cases.csv"

        case_df = DataFrame(CSV.File(CSVpath, transpose=transposed_df))

        times = case_df.time

        if transposed_df
            case_df = select(case_df, 2:ncol(case_df))
        end

        dailyConfirmedCases = zeros(length(times), Int64((ncol(case_df)-1) /2))
        dailyTotalCases = zeros(length(times), Int64((ncol(case_df)-1) /2))

        for col in 1:length(dailyConfirmedCases[1,:])
            colname = "$type confirmed cases - run_$col"
            dailyConfirmedCases[:,col] = case_df[:, colname]

            colname = "$type total cases - run_$col"
            dailyTotalCases[:,col] = case_df[:, colname]

        end

        return times, dailyConfirmedCases, dailyTotalCases
    end

    function reloadCSVmodels(CSVpath::String)::DataFrame

        models_df = DataFrame(CSV.File(CSVpath))

        return models_df
    end

    function removeNaNs(array::Array{Float64,2}, newVal=0.0)
        #=
        Replace all NaNs in a 2d array with 0.0
        =#

        for i in 1:length(array[1,:])
            for j in 1:length(array[:,1])
                if isnan(array[j,i])
                    array[j,i]=newVal
                end
            end
        end

        return array
    end

end  # module outbreakPostProcessing
