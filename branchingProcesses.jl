using Distributions
using Random
using DataFrames
using StatsBase
using LightGraphs, GraphPlot, NetworkLayout
using BenchmarkTools
using TrackingHeaps
using DataStructures

# import required modules
push!( LOAD_PATH, "./" )
using plotsPyPlot: plotBranchPyPlot
using BranchVerifySoln: branchVerifyPlot, meanAbsError, initSIRArrays, multipleSIRMeans,
    multipleLinearSplines, branchTimeStepPlot

function weibullParGridSearch(alpha_par, scale_par)

    # alpha_par = 2.86 #2.83
    # scale_par = 5.62 #5.67

    # mean = θ * gamma(1 + 1/α)
    # sd = sqrt(d.θ^2 * gamma(1 + 2/d.α) - mean^2)

    # a = Weibull(alpha_par, scale_par)

    # mean(a) # ~ 5.0

    # sqrt(var(a)) # ~ 1.9

    # set up grid search
    delta = (cumsum(ones(1000)).-500)*0.0001

    alpha_range = alpha_par .+ delta
    scale_range = scale_par .+ delta

    minDistance = Inf;
    a_best = 0;
    s_best = 0;

    for a in alpha_range
        for s in scale_range
            dist = Weibull(a,s)

            distance = abs(mean(dist)-5.0) + abs(sqrt(var(dist))-1.9)

            if distance < minDistance
                minDistance = distance * 1
                a_best = a*1
                s_best = s*1
            end

        end
    end;

    mean(Weibull(a_best, s_best))
    sqrt(var(Weibull(a_best, s_best)))
end

mutable struct branchModel
    t_init::Number
    t_max::Number

    population_size::Int
    max_cases::Int
    state_totals::Array{Int64,1} # first value will be "S", second "I", third "R"
    states::Array{String,1}

    sub_clin_prop::Float64
    sub_clin_scaling::Float64

    reproduction_number::Number
    reproduction_k::Float64 # Superspreading k a la Lloyd-Smith
    stochasticRi::Bool

    # Weibull distribution for time to an infection event being caused by an
    # active individual
    t_generation_shape::Float64 #
    t_generation_scale::Float64

    ###### isolation parameters to add later ####
    t_onset_shape::Float64
    t_onset_scale::Float64
    t_onset_to_isol::Float64
    stochasticIsol::Bool

    p_test::Float64 # ∈ [0,1]

    #############################################

    recovery_time::Float64 # time taken for recovery (no randomness in this)

    isolation_infectivity::Number # take to be zero
end

struct event
    time::Union{Float64, Int64}
    isRecovery::Bool # false is infection event, true is recovery event
    parentID::Int64
    sSaturation::Float64 # ∈ [0,1]
end

function Base.isless(x::event, y::event)::Bool
    return  x.time < y.time
end

function getRi(model::branchModel, sub_Clin_Case::BitArray{1})
    #=
    Reproduction number is scaled by a gamma distribution
    =#

    Ri = ones(model.max_cases)

    if model.stochasticRi
        Ri .= rand(Gamma(1/model.reproduction_k, model.reproduction_k), model.max_cases)
    end

    # Scale by reproduction number and whether case is sub clinical or not
    Ri = Ri .* model.reproduction_number .* (1 .- model.sub_clin_scaling.*sub_Clin_Case)

    return Ri
end

function getOnsetDelay(model::branchModel)
    #=
    Onset delay for symptoms is gamma distributed if using stochastic sim
    Otherwise it is the mean of the gamma distribution defined
    =#

    if model.stochasticIsol
        return rand(Gamma(model.t_onset_shape, model.t_onset_scale), model.max_cases)
    end

    return zeros(model.max_cases) .+ mean(Gamma(model.t_onset_shape, model.t_onset_scale))
end

function getOnsetToIsolDelay(model::branchModel, num_rand)
    #=
    Delay between symptom onset and isolation is exponentially distributed if using
    stochastic sim. Otherwise it is the mean of the exponential distribution defined
    =#
    if model.stochasticIsol
        return rand(Exponential(model.t_onset_to_isol), num_rand)
    end
    return zeros(num_rand) .+ model.t_onset_to_isol
end

function getTimeIsolated(model::branchModel, detected_cases::Union{SubDataFrame,DataFrameRow})
    #=
    The time that a clinical case is isolated, if they are detected
    =#
    return detected_cases.time_infected .+ detected_cases.time_onset_delay .+
        getOnsetToIsolDelay(model, length(detected_cases.active))
end

function initDataframe(model::branchModel)

    population_df = DataFrame()

    # dataframe columns init
    parentID = Array{Int64}(undef, model.max_cases) .* 0
    parentID[1:model.state_totals[2]] .= -1 # they are a root node

    num_offspring = Array{Int64}(undef, model.max_cases) .* 0

    # arguably this is just the row value so technically unneeded
    caseID = convert.(Int,cumsum(ones(model.max_cases)))

    # numOff - technically unneeded

    active = convert.(Bool,zeros(model.max_cases))
    active[1:model.state_totals[2]] .= true

    time_infected = Array{Float64}(undef, model.max_cases) .* 0.0
    time_infected[1:model.state_totals[2]] .= model.t_init * 1.0

    # init whether a given case is subClin or not.
    sub_Clin_Case = rand(model.max_cases) .< model.sub_clin_prop

    reproduction_number = getRi(model, sub_Clin_Case)

    time_onset_delay = getOnsetDelay(model)
    time_isolated = ones(model.max_cases) .* Inf

    population_df.parentID = parentID
    population_df.caseID = caseID
    population_df.active = active
    population_df.sub_Clin_Case = sub_Clin_Case
    population_df.reproduction_number = reproduction_number
    population_df.num_offspring = num_offspring
    population_df.time_infected = time_infected
    population_df.time_recovery = active .* model.recovery_time
    population_df.time_onset_delay = time_onset_delay
    population_df.time_isolated = time_isolated

    population_df.detected = convert.(Bool,zeros(model.max_cases))
    clin_cases = filter(row -> row.sub_Clin_Case==false, population_df, view=true)
    clin_cases.detected .= rand(Bernoulli(model.p_test), length(clin_cases.active))

    if model.p_test > 0
        detected_cases = filter(row -> row.detected==true, clin_cases, view=true)
        detected_cases.time_isolated .= getTimeIsolated(model, detected_cases)
    end

    # need to init isolation times for these guys^


    return population_df
end

function simpleGraph_branch(population_df::DataFrame, max_num_nodes, single_root::Bool, root_node=1)
    #=
    Construct a infection tree, beginning with initial cases, containing
    min(max_num_nodes, num_cases in population_df)
    =#

    num_cases = length(population_df.caseID)
    graph_size = minimum([max_num_nodes, num_cases])

    # plot the entire infection tree
    if !single_root
        infection_tree = SimpleDiGraph(graph_size)

        for i in 1:graph_size
            add_edge!(infection_tree, population_df[i, :parentID], population_df[i, :caseID])
        end

        gplot(infection_tree, arrowlengthfrac=0.02)

    else # plot only the infection tree that begins at a given root node
        infection_tree = SimpleDiGraph()
        parents_in_graph = convert.(Bool, zeros(graph_size))

        parents_in_graph[root_node] = true
        add_vertex!(infection_tree)
        vertex_number = 1
        vertex_label_dict = Dict(root_node=>vertex_number)

        for i in 1:graph_size
            if population_df[i, :parentID] == -1
                # do nothing
            elseif parents_in_graph[population_df[i, :parentID]]
                add_vertex!(infection_tree)

                vertex_number += 1
                vertex_label_dict[population_df[i, :caseID]] = vertex_number

                add_edge!(infection_tree, vertex_label_dict[population_df[i, :parentID]], vertex_number)

                parents_in_graph[population_df[i, :caseID]] = true
            end
        end

        gplot(infection_tree, arrowlengthfrac=0.02)
    end
end

function randRootNode(population_df::DataFrame)
    # choose a root node to graph
    root_df = filter(row -> row.parentID == -1 && row.num_offspring>0, population_df, view=true)

    return rand(root_df.caseID)
end

function initTime(model::branchModel, time_step::Number)
    #=
    (if time step is 1 day)
    Time step 1 refers to time period 0->1
    Time step 2 refers to time period 1->2

    So time step 50 refers to period 49->50 and is the final step (if reached)
    =#

    # t = Float64[copy(model.t_init)]

    # if timestep does not divide perfectly, stop before max_time
    # however we will run sims only where it does.

    # time span to solve on
    tspan = (model.t_init,model.t_max)

    # times to solve on
    t = [i for i=tspan[1]:time_step:tspan[end]]

    num_steps = convert(Int, floor((model.t_max - model.t_init) / time_step))

    # t = zeros(num_steps+1)
    #
    # t[1] = model.t_init
    # t[2:end] .= t[1] .+ cumsum(ones(num_steps)) .* time_step

    return t::Array{}, num_steps::Int
end

function initTime_infection(model::branchModel, time_step::Number)
    num_steps = convert(Int, ceil(model.recovery_time / time_step))

    t = zeros(num_steps+1)
    t[2:end] .= collect(1:num_steps) .* time_step

    return t::Array{}
end

function initStateTotals(model::branchModel, times_length::Int64)::Array{Int64, 2}
    #=
    initialise a 2d array of state totals for each discrete time
    =#

    state_totals_all = convert.(Int64, zeros(times_length, length(model.state_totals)))
    state_totals_all[1,:] .= copy(model.state_totals)

    return state_totals_all
end

function makeVectorFromFrequency(f::Union{SubArray{Int64,1},Array{Int64,1}}, ID::Union{SubArray{Int64,1},Array{Int64,1}})::Array{Int64,1}
    #=
    Code from Michael Plank with some edits for my chosen code structure

    Generates a vector v of integers in non-descending order such that the number of occurrences of k is f(k)
    Integers are between 1 and length(f), and the number of elements of v is sum(f)

    INPUTS: f - column vector of required frequencies of integers 1 to N where N =
    length(f). These integers will refer to the IDs stored in the integer positions
            ID - column of IDs
    OUTPUTS: v - column vector of integers

    USAGE:  v = makeVectorFromFreq(f)

    EXAMPLE: v = makeVectorFromFrequency([0, 0, 1, 0, 0, 2, 0, 2], [1, 2, 3, 4, 5, 6, 7, 8])
       returns v = [3 6 6 8 8]
    =#
    count = sum(f)

    v = convert.(Int64, zeros(count))

    index = 1
    IDIndex = 1
    for i in f
        for j in 1:i
            v[index] = ID[IDIndex]
            index += 1
        end
        IDIndex +=1
    end

    return v
end

function initNewCases!(population_df::DataFrame, active_df::SubDataFrame,
    model::branchModel, t::Union{Array{Float64,1},Array{Int64,1}}, current_step::Int, num_cases::Int,
    num_new_infections::Int, num_off::Array{})
    #=
    Initialises all new cases that occured within a given time step

    Works in place on the active dataframe
    =#

    new_cases_rows = @view population_df[num_cases+1:num_cases+num_new_infections,:]
    new_cases_rows.parentID .= makeVectorFromFrequency(num_off, active_df.caseID)
    new_cases_rows.active .= true
    new_cases_rows.time_infected .= t[current_step]
    new_cases_rows.time_recovery .= t[current_step] + model.recovery_time

    if model.p_test > 0
        detected_cases = filter(row -> row.detected==true, new_cases_rows, view=true)
        detected_cases.time_isolated .= getTimeIsolated(model, detected_cases)
    end

    active_df[:,:num_offspring] .+= num_off

    model.state_totals[1] -= num_new_infections
    model.state_totals[2] += num_new_infections
end

function initNewCase!(population_df::DataFrame, model::branchModel, infection_time::Union{Float64,Int64}, parentID::Int64, num_cases::Int64)
    #=
    Initialises the new case that just occured for the first react branch

    Works in place on the population dataframe
    =#

    # else, set the next case to active with relevant parentID, infectionTime
    # and increment num_offspring for case that did infection

    model.state_totals[1] -=1
    model.state_totals[2] +=1

    new_case_row = @view population_df[num_cases, :]
    new_case_row.parentID = parentID * 1
    new_case_row.active = true
    new_case_row.time_infected = infection_time * 1
    new_case_row.time_recovery = infection_time + model.recovery_time

    if new_case_row.detected
        new_case_row.time_isolated = getTimeIsolated(model, new_case_row)[1]
    end

    population_df[parentID, :num_offspring] += 1
end

function recovery_branch!(population_df::DataFrame, model::branchModel, caseID::Int64)
    #=
    Performs a recovery on the given individual. Sets them inactive and increments
    totals

    Works in place on the population dataframe
    =#
    population_df[caseID, :active] = false
    model.state_totals[2] -=1
    model.state_totals[3] +=1
end

function areaUnderCurve(area_under_curve::Array{Float64,1}, index::Array{Int64,1})::Array{Float64,1}

    area_under_curve_i = index .* 0.0

    for i::Int in 1:length(index)
        area_under_curve_i[i] = area_under_curve[index[i]]
    end

    return area_under_curve_i
end

function initDataframe_thin(model::branchModel)

    population_df = DataFrame()

    # dataframe columns init
    parentID = Array{Int64}(undef, model.max_cases) .* 0
    parentID[1:model.state_totals[2]] .= -1 # they are a root node

    # num_offspring = Array{Int64}(undef, model.max_cases) .* 0

    # arguably this is just the row value so technically unneeded
    caseID = convert.(Int,cumsum(ones(model.max_cases)))

    generation_number = Array{Int64}(undef, model.max_cases) .* 0

    # init whether a given case is subClin or not.
    sub_Clin_Case = rand(model.max_cases) .< model.sub_clin_prop

    reproduction_number = getRi(model, sub_Clin_Case)

    population_df.parentID = parentID
    population_df.caseID = caseID
    population_df.generation_number = generation_number
    population_df.reproduction_number = reproduction_number
    # population_df.num_offspring = num_offspring

    return population_df
end

function branchingProcess!(population_df::DataFrame, model::branchModel)
    generation::Int64 = 1

    num_cases::Int64 = model.state_totals[2]*1
    generationRange = 1:num_cases
    newGenerationRange = [generationRange[1], generationRange[2]]
    population_df[generationRange, :generation_number] .= copy(generation)

    population_df.num_offspring = rand.(Poisson.(population_df.reproduction_number))
    hitMaxCases = false

    while num_cases < model.max_cases

        # determine number of offspring
        num_off = @view population_df[generationRange, :num_offspring]

        total_off = sum(num_off)
        if total_off == 0
            break
        end

        # new generation
        generation+=1
        newGenerationRange = [num_cases+1, total_off+num_cases]

        ########## Logic from discrete
        if newGenerationRange[2]>model.max_cases >= model.max_cases
            hitMaxCases = true
            newGenerationRange[2] = model.max_cases

            amount_above_limit = num_cases + total_off - model.max_cases

            # allow up to the model's limit's cases to be added
            total_off = model.max_cases - num_cases

            if amount_above_limit > 0
                for i::Int64 in 1:length(num_off::SubArray{Int64,1})

                    if num_off[i] > amount_above_limit
                        num_off[i] -= amount_above_limit
                        amount_above_limit = 0
                    else
                        amount_above_limit -= num_off[i]
                        num_off[i]=0
                    end

                    if amount_above_limit == 0
                        break
                    end
                end
            end
        end
        #######################

        range = newGenerationRange[1]:newGenerationRange[2]

        population_df[range, :generation_number] .= copy(generation)
        population_df[range, :parentID] .= makeVectorFromFrequency(num_off, collect(generationRange))

        num_cases += total_off
        if hitMaxCases
            population_df[range, :num_offspring] .= 0
        end

        generationRange = copy(range)

        # try
        # catch LoadError
        #     println(length(population_df[newGenerationRange, :parentID]))
        #     println(length(makeVectorFromFrequency(num_off, collect(generationRange))))
        #     break
        # end

    end
end

function discrete_branch(population_df::DataFrame, model::branchModel, time_step::Number)
    #=
    A discrete branching process based off of Matlab code by Michael Plank

    For simplicity, it's recommended that this is used with time steps of 1 day
    or multiples/divisors of days.

    (if time step is 1 day, t_init = 0, t_max = 50)
    Time step 1 refers to time period 0->1
    Time step 2 refers to time period 1->2

    So time step 50 refers to period 49->50 and is the final step (if reached)

    =#

    num_cases = model.state_totals[2]*1
    infection_time_dist = Weibull(model.t_generation_shape, model.t_generation_scale)

    t_infection = initTime_infection(model, time_step)
    area_under_curve = diff(cdf.(infection_time_dist, t_infection))

    t, num_steps = initTime(model, time_step)
    state_totals_all = initStateTotals(model, length(t))

    current_step = 1
    # filter df on active infections
    active_df = filter(row -> row.active, population_df, view=true)
    hitMaxCases::Bool = (num_cases >= model.max_cases)

    for current_step::Int in 2:(num_steps+1)

        # make inactive any individuals whose time of recovery occurs during or by end of time step
        if t[current_step] >= model.recovery_time

            inactive_df = filter(row-> row.time_recovery < t[current_step], active_df, view=true)

            num_recovered = length(inactive_df.active)
            if num_recovered > 0
                inactive_df.active .= false
                model.state_totals[2] -= num_recovered
                model.state_totals[3] += num_recovered
            end
        end

        # filter df on active infections
        active_df = filter(row -> row.active, population_df, view=true)

        # determine which cases are isolating (if any)
        case_isolated = active_df.time_isolated .< t[current_step]

        # if reached max cases / other stopping criterion
        # MAX CASE CRITERION WILL NEVER GET HIT - WILL BREAK DATAFRAME FIRST
        if hitMaxCases || model.state_totals[2] == 0

            # simulation is over
            for i in current_step:(num_steps+1)
                state_totals_all[i,:] = state_totals_all[current_step-1,:]
            end
            break
        end

        # Determine number offspring for each active individual in current_step
        exp_off = (model.state_totals[1]/model.population_size) .*
            active_df.reproduction_number .* (1 .- (1-model.isolation_infectivity).*case_isolated) .*
            areaUnderCurve(area_under_curve, convert.(Int64, round.((t[current_step].-active_df.time_infected)/time_step)))

        num_off = rand.(Poisson.(exp_off))
        num_new_infections = sum(num_off)

        if num_new_infections > 0

            # move this code into a function #######
            if num_cases + num_new_infections >= model.max_cases
                hitMaxCases = true

                amount_above_limit = num_cases + num_new_infections - model.max_cases

                # allow up to the model's limit's cases to be added
                num_new_infections = model.max_cases - num_cases

                if amount_above_limit > 0
                    for i::Int in 1:length(num_off)

                        if num_off[i] > amount_above_limit
                            num_off[i] -= amount_above_limit
                            amount_above_limit = 0
                        else
                            amount_above_limit -= num_off[i]
                            num_off[i]=0
                        end

                        if amount_above_limit == 0
                            break
                        end
                    end
                end
            end
            ###########################################

            initNewCases!(population_df, active_df, model, t, current_step, num_cases, num_new_infections, num_off)
            num_cases += num_new_infections
        end

        state_totals_all[current_step, :] .= copy(model.state_totals)
    end

    return t, state_totals_all, num_cases
end

function firstReact_branch(population_df::DataFrame, model::branchModel)

    num_cases = model.state_totals[2]*1
    num_events = 1
    t = Float64[copy(model.t_init)]

    state_totals_all = initStateTotals(model, 1+(model.max_cases-num_cases)*2)
    infection_time_dist = Weibull(model.t_generation_shape, model.t_generation_scale)

    while t[end] < model.t_max && model.state_totals[2] != 0 && num_cases < model.max_cases

        # filter df on active infections
        active_df = filter(row -> row.active, population_df[1:num_cases, :], view=true)

        active_time_left = active_df.time_recovery .- t[end]
        active_time_spent = round.(model.recovery_time .- active_time_left, digits=9)

        # Draw number of reactions for each active infected individual
        # expected number of reactions in time remaining, given time distribution
        expOff = (model.state_totals[1]/model.population_size) .*
            active_df.reproduction_number .* ccdf.(infection_time_dist, t[end].-active_df.time_infected)

        num_off = rand.(Poisson.(expOff))

        if !isa(num_off, Array)
            num_off = [num_off]
        end

        # find minimum time to any reaction for each active individual (including recovery)
        min_time = num_off .* 0.0

        Threads.@threads for i::Int64 in 1:length(min_time)
            if num_off[i] == 0
                min_time[i] = active_time_left[i]
            else
                # need to do inverse sampling based on time since infection occurred
                cdf_at_time = cdf(infection_time_dist, active_time_spent[i])

                ###### multiple samples for each offspring
                log_transformed_rand = log.(rand(num_off[i]) .* (1-cdf_at_time) .+ cdf_at_time)

                timeToReact = invlogcdf.(infection_time_dist, log_transformed_rand) .- active_time_spent[i]

                min_time[i] = minimum([minimum(timeToReact), active_time_left[i]])
                #################

                # ####### one sample only
                # log_transformed_rand = log(rand() * (1-cdf_at_time) + cdf_at_time)
                #
                # timeToReact = invlogcdf(infection_time_dist, log_transformed_rand) - active_time_spent[i]
                #
                # min_time[i] = minimum([timeToReact, active_time_left[i]])
            end
        end

        # find overall minimum time
        active_index = argmin(min_time)
        infection_time = t[end] + min_time[active_index]
        ID = active_df[active_index, :caseID]

        # recovery event
        if abs(population_df[ID, :time_recovery] - infection_time) < 10^-8
            recovery_branch!(population_df, model, ID)

        else # infection event
            num_cases += 1
            initNewCase!(population_df, model, infection_time, ID, num_cases)
        end

        num_events += 1
        push!(t, infection_time*1)
        state_totals_all[num_events, :] .= copy(model.state_totals)
    end

    if t[end] < model.t_max
        num_events += 1
        push!(t, model.t_max)
        state_totals_all[num_events, :] .= copy(model.state_totals)
    end

    return t, state_totals_all[1:num_events,:], num_cases
end

function nextReact_branch_trackedHeap(population_df::DataFrame, model::branchModel)

    num_cases::Int64 = model.state_totals[2]*1
    num_events = 1
    t = Float64[copy(model.t_init)]

    state_totals_all = initStateTotals(model, 1+(model.max_cases-num_cases)*2)
    infection_time_dist = Weibull(model.t_generation_shape, model.t_generation_scale)

    # filter df on active infections
    active_df = filter(row -> row.active, population_df[1:num_cases, :], view=true)

    num_tau_events = model.max_cases * (model.reproduction_number*1.2 + 1)
    tau_i = [event(model.t_max+i,false,0,0) for i=1:num_tau_events]
    tau_heap = TrackingHeap(event, S=NoTrainingWheels, O=MinHeapOrder, N = 2, init_val_coll=tau_i)

    next_unused_index = 1
    sSaturation = (model.state_totals[1]/model.population_size)

    expOff = (model.state_totals[1]/model.population_size) .* active_df.reproduction_number
    num_off = rand.(Poisson.(expOff))

    for i in 1:length(active_df.time_infected)
        # insert recovery events
        update!(tau_heap, next_unused_index, event(active_df[i,:time_recovery], true, active_df[i,:caseID], sSaturation))
        next_unused_index += 1

        # insert infection events
        for j in 1:num_off[i]
            update!(tau_heap, next_unused_index, event(t[end]+rand(infection_time_dist), false, active_df[i,:caseID], sSaturation))
            next_unused_index += 1
        end
    end


    while t[end] < model.t_max && model.state_totals[2] != 0 && num_cases < model.max_cases

        # returns a pair
        reaction = pop!(tau_heap)
        infection_time = reaction[2].time

        if reaction[2].isRecovery
            recovery_branch!(population_df, model, reaction[2].parentID)
            # pop!(tau_heap)

            num_events += 1
            state_totals_all[num_events, :] .= copy(model.state_totals)
            push!(t, infection_time*1)

        else # infection event

            sSaturation = (model.state_totals[1]/model.population_size)
            # rejection step for population saturation
            if rand() < sSaturation/reaction[2].sSaturation

                # rejection step for detection and subsequent isolation (short circuit OR)
                if !population_df[reaction[2].parentID,:detected] ||
                    infection_time < population_df[reaction[2].parentID,:time_isolated] ||
                    rand() < model.isolation_infectivity

                    num_cases += 1
                    initNewCase!(population_df, model, infection_time, reaction[2].parentID, num_cases)

                    update!(tau_heap, next_unused_index, event(population_df[num_cases,:time_recovery], true, population_df[num_cases,:caseID], sSaturation))
                    next_unused_index += 1

                    # determine num offspring
                    expOff = sSaturation * population_df[num_cases, :reproduction_number]
                    num_off = rand(Poisson(expOff))

                    # new infections
                    for j in 1:num_off
                        update!(tau_heap, next_unused_index, event(infection_time+rand(infection_time_dist), false, population_df[num_cases,:caseID], sSaturation))
                        next_unused_index += 1
                    end

                    num_events += 1
                    state_totals_all[num_events, :] .= copy(model.state_totals)
                    push!(t, infection_time*1)
                end
            # else
                # get rid of infection event (didn't happen)
                # pop!(tau_heap)

            end
        end

    end

    if t[end] < model.t_max
        num_events += 1
        push!(t, model.t_max)
        state_totals_all[num_events, :] .= copy(model.state_totals)
    end

    return t, state_totals_all[1:num_events,:], num_cases
end

function nextReact_branch(population_df::DataFrame, model::branchModel)

    num_cases::Int64 = model.state_totals[2]*1
    num_events = 1
    t = Float64[copy(model.t_init)]

    state_totals_all = initStateTotals(model, 1+(model.max_cases-num_cases)*2)
    infection_time_dist = Weibull(model.t_generation_shape, model.t_generation_scale)

    # filter df on active infections
    active_df = filter(row -> row.active, population_df[1:num_cases, :], view=true)

    # num_tau_events = model.max_cases * (model.reproduction_number*1.2 + 1)
    # tau_i = [event(model.t_max+i,false,0,0) for i=1:num_tau_events]
    tau_heap = MutableBinaryHeap{event, DataStructures.FasterForward}()

    sSaturation = (model.state_totals[1]/model.population_size)

    expOff = (model.state_totals[1]/model.population_size) .* active_df.reproduction_number
    num_off = rand.(Poisson.(expOff))

    for i in 1:length(active_df.time_infected)
        # insert recovery events
        push!(tau_heap, event(active_df[i,:time_recovery], true, active_df[i,:caseID], sSaturation))

        # insert infection events
        for j in 1:num_off[i]
            push!(tau_heap, event(t[end]+rand(infection_time_dist), false, active_df[i,:caseID], sSaturation))
        end
    end


    while t[end] < model.t_max && model.state_totals[2] != 0 && num_cases < model.max_cases

        # returns a event
        reaction = pop!(tau_heap)
        infection_time = reaction.time

        if reaction.isRecovery
            recovery_branch!(population_df, model, reaction.parentID)
            # pop!(tau_heap)

            num_events += 1
            state_totals_all[num_events, :] .= copy(model.state_totals)
            push!(t, infection_time*1)

        else # infection event

            sSaturation = (model.state_totals[1]/model.population_size)
            # rejection step for population saturation
            if rand() < sSaturation/reaction.sSaturation

                # rejection step for detection and subsequent isolation (short circuit OR)
                if !population_df[reaction.parentID,:detected] ||
                    infection_time < population_df[reaction.parentID,:time_isolated] ||
                    rand() < model.isolation_infectivity

                    num_cases += 1
                    initNewCase!(population_df, model, infection_time, reaction.parentID, num_cases)

                    push!(tau_heap, event(population_df[num_cases,:time_recovery], true, population_df[num_cases,:caseID], sSaturation))

                    # determine num offspring
                    expOff = sSaturation * population_df[num_cases, :reproduction_number]
                    num_off = rand(Poisson(expOff))

                    # new infections
                    for j in 1:num_off
                        push!(tau_heap, event(infection_time+rand(infection_time_dist), false, population_df[num_cases,:caseID], sSaturation))
                    end

                    num_events += 1
                    state_totals_all[num_events, :] .= copy(model.state_totals)
                    push!(t, infection_time*1)
                end
            end
        end

    end

    if t[end] < model.t_max
        num_events += 1
        push!(t, model.t_max)
        state_totals_all[num_events, :] .= copy(model.state_totals)
    end

    return t, state_totals_all[1:num_events,:], num_cases
end


function init_model_pars(t_init::Number, t_max::Number, population_size::Int, max_cases::Int, state_totals)::branchModel

    # t_init=
    # t_max::Number

    # population_size::Int
    # max_cases::Int
    # state_totals::Array{Int64,1} # first value will be "S", second "I", third "R"
    states = ["S", "I", "R"]

    sub_clin_prop = 1/3
    sub_clin_scaling = 0.5

    reproduction_number = 3
    reproduction_k = 0.5 # Superspreading k a la Lloyd-Smith
    stochasticRi = true

    # Weibull distributed
    t_generation_shape = 2.826
    t_generation_scale = 5.665

    ###### isolation parameters to add later ####
    # gamma distributed
    t_onset_shape = 5.8
    t_onset_scale = 0.95

    # Exponentially distributed
    t_onset_to_isol = 2.2
    stochasticIsol = true
    p_test = 0#0.75

    #############################################

    recovery_time = 30 # time taken for recovery (no randomness in this)

    isolation_infectivity = 0 # take to be zero

    model = branchModel(t_init, t_max, population_size, max_cases, state_totals, states,
        sub_clin_prop, sub_clin_scaling, reproduction_number, reproduction_k, stochasticRi,
        t_generation_shape, t_generation_scale, t_onset_shape, t_onset_scale, t_onset_to_isol,
        stochasticIsol, p_test, recovery_time, isolation_infectivity)

    return model
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
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^8, 10000, [5*10^8-10,10,0]);
            model.stochasticRi = false
            model.sub_clin_prop = 0

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)

            reproduction_number = model.reproduction_number
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $reproduction_number")

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of First Case")
        numSims = convert(Int, round(40 / numSimsScaling))

        meanOffspring = zeros(numSims)
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^8, 3000, [5*10^8-10,10,0]);
            model.stochasticRi = false
            model.sub_clin_prop = 0
            model.recovery_time = 20

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = firstReact_branch(population_df, model)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)

            reproduction_number = model.reproduction_number
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $reproduction_number")
        println("Finished Simulation in $time seconds")
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
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^8, 10000, [5*10^8-10,10,0]);
            model.stochasticRi = true
            model.sub_clin_prop = 0

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            reproduction_number = model.reproduction_number
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $reproduction_number")

        println("Finished Simulation in $time seconds")

        println("Beginning simulation of First Case")
        numSims = convert(Int, round(40 / numSimsScaling))

        meanOffspring = zeros(numSims)
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^8, 3000, [5*10^8-10,10,0]);
            model.stochasticRi = true
            model.sub_clin_prop = 0
            model.recovery_time = 20

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = firstReact_branch(population_df, model)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            reproduction_number = model.reproduction_number
        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $reproduction_number")
        println("Finished Simulation in $time seconds")
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
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^8, 10000, [5*10^8-10,10,0]);
            model.stochasticRi = false
            model.sub_clin_prop = 0.5

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^8, 3000, [5*10^8-10,10,0]);
            model.stochasticRi = false
            model.sub_clin_prop = 0.5
            model.recovery_time = 20

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = firstReact_branch(population_df, model)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            meanRNumber[i] = mean(inactive_df.reproduction_number)

        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $(mean(meanRNumber))")
        println("Finished Simulation in $time seconds")
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
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^8, 10000, [5*10^8-10,10,0]);
            model.stochasticRi = true
            model.sub_clin_prop = 0.5

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^8, 3000, [5*10^8-10,10,0]);
            model.stochasticRi = true
            model.sub_clin_prop = 0.5
            model.recovery_time = 20

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = firstReact_branch(population_df, model)

            inactive_df = filter(row -> row.active==false && row.parentID!=0, population_df[1:num_cases, :], view=true)

            meanOffspring[i] = mean(inactive_df.num_offspring)
            meanRNumber[i] = mean(inactive_df.reproduction_number)

        end

        println("Mean actual offspring was $(mean(filter!(!isnan, meanOffspring))), for reproduction number of $(mean(meanRNumber))")
        println("Finished Simulation in $time seconds")
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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

            # interpolate using linear splines
            StStep[:,i], ItStep[:,i], RtStep[:,i] = multipleLinearSplines(state_totals_all, t, times)
        end

        println("Finished Simulation in $time seconds")

        Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

        discreteSIR_mean = hcat(Smean, Imean, Rmean)


        println("Beginning simulation of First Case")
        numSims = convert(Int, round(40 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = firstReact_branch(population_df, model)

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==model.recovery_time,t)
            lastDupe = findlast(x->x==model.recovery_time,t)

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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = nextReact_branch(population_df, model)

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==model.recovery_time,t)
            lastDupe = findlast(x->x==model.recovery_time,t)

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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = nextReact_branch(population_df, model)

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==model.recovery_time,t)
            lastDupe = findlast(x->x==model.recovery_time,t)

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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            model.p_test = 1.0
            model.sub_clin_prop = 0
            model.stochasticIsol = false
            # model.t_onset_shape = 5.8
            model.t_onset_to_isol = 0

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            model.p_test = 1.0
            model.sub_clin_prop = 0
            model.stochasticIsol = false
            # model.t_onset_shape = 5.8
            model.t_onset_to_isol = 0

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = nextReact_branch(population_df, model)

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==model.recovery_time,t)
            lastDupe = findlast(x->x==model.recovery_time,t)

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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            model.p_test = 1.0
            model.sub_clin_prop = 0
            model.stochasticIsol = false
            # model.t_onset_shape = 5.8
            model.t_onset_to_isol = 0

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            model.p_test = 1.0
            model.sub_clin_prop = 0
            model.stochasticIsol = false
            # model.t_onset_shape = 5.8
            model.t_onset_to_isol = 0

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = nextReact_branch(population_df, model)

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==model.recovery_time,t)
            lastDupe = findlast(x->x==model.recovery_time,t)

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
    end
end

function discreteSIR_sim(time_step::Union{Float64, Int64}, numSimulations::Int64, tspan, numSimsScaling)

    # times to sim on
    times = [i for i=tspan[1]:time_step:tspan[end]]

    numSims = convert(Int, round(numSimulations / numSimsScaling))

    StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

    i = 1
    time = @elapsed Threads.@threads for i = 1:numSims

        model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

        population_df = initDataframe(model);
        t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

        StStep[:,i] = state_totals_all[:,1]
        ItStep[:,i] = state_totals_all[:,2]
        RtStep[:,i] = state_totals_all[:,3]

    end

    println("Finished Simulation in $time seconds")

    Smean, Imean, Rmean = multipleSIRMeans(StStep, ItStep, RtStep)

    return hcat(Smean, Imean, Rmean), times
end

function compilationInit()
    # discrete
    model = init_model_pars(0, 100, 5*10^3, 5*10^3, [5*10^3-10,10,0]);
    population_df = initDataframe(model);
    time_step = 1;
    @time t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)

    # first
    model = init_model_pars(0, 20, 5*10^7, 100, [5*10^7-10,10,0])
    population_df = initDataframe(model)
    @time t, state_totals_all, num_cases = firstReact_branch(population_df, model)

    # next
    model = init_model_pars(0, 100, 5*10^3, 5*10^3, [5*10^3-10,10,0])
    population_df = initDataframe(model);
    @time t, state_totals_all, num_cases= nextReact_branch(population_df, model)
end

compilationInit()

# verifySolutions(1, [[5,6,7,9,10]])

model = init_model_pars(0, 200, 5*10^6, 5*10^6, [5*10^6-10,10,0])
time_step = 1;
# model.p_test = 1.0
# model.sub_clin_prop = 0
# model.stochasticIsol = false
# model.t_onset_shape = 5.8
# model.t_onset_to_isol = 0

population_df = initDataframe(model);
@time t, state_totals_all, num_cases = discrete_branch(population_df, model, time_step)
outputFileName = "juliaGraphs/branchDiscrete/branch_model_$(model.population_size)"
# subtitle = "Discrete model with timestep of $time_step days"
# plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)

# # next tracked heap
model = init_model_pars(0, 200, 5*10^6, 5*10^3, [5*10^6-10,10,0]);
population_df = initDataframe(model);
@time t, state_totals_all, num_cases = nextReact_branch_trackedHeap(population_df, model)
outputFileName = "juliaGraphs/branchNextReact/branch_model_$(model.population_size)"
subtitle = "Next react model"
plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)

# # next, regular heap
model = init_model_pars(0, 200, 5*10^6, 5*10^3, [5*10^6-10,10,0]);
population_df = initDataframe(model);
@time t, state_totals_all, num_cases = nextReact_branch(population_df, model)
outputFileName = "juliaGraphs/branchNextReact/branch_model_$(model.population_size)"
subtitle = "Next react model"
plotBranchPyPlot(t, state_totals_all, model.population_size, outputFileName, subtitle, true, false)




model = init_model_pars(0, 200, 5*10^6, 5*10^6, [5*10^6-10,10,0]);
population_df = initDataframe_thin(model);
@time branchingProcess!(population_df, model)

# print(population_df)

simpleGraph_branch(population_df, 1000, true, 2)

population_df
