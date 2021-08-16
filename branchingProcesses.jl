using DataFrames
using BenchmarkTools
using Distributions, Random, StatsBase
using LightGraphs, GraphPlot, NetworkLayout
using TrackingHeaps, DataStructures, SparseArrays

# import required modules
push!( LOAD_PATH, "./" )
using plotsPyPlot: plotBranchPyPlot, plotSimpleBranchPyPlot
using BranchVerifySoln: branchVerifyPlot, meanAbsError, initSIRArrays, multipleSIRMeans,
    multipleLinearSplines, branchTimeStepPlot, branch2wayVerifyPlot, Dots, Lines

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

struct ThinNone; end
struct ThinSingle; end
struct ThinTree; end
struct ThinFirst; end

struct ThinFunction
    name::Union{ThinNone, ThinSingle, ThinTree, ThinFirst}
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

mutable struct Node
    caseID::Int64
    infection_time::Union{Float64,Int64}
    parent::Union{Base.RefValue{Node},Base.RefValue{Nothing}}
    children::Array{Node,1}
end

struct Tree
    rootNodes::Array{Node,1}
end

function Base.isequal(x::Node, y::Node); x.caseID === y.caseID; end

function deleteChildNode!(childNode::Node)
    #=
    Given a child node, delete the child from it's parent's children array
    =#

    parent = childNode.parent[]
    for i in 1:length(parent.children)
        if isequal(parent.children[i], childNode)
            deleteat!(parent.children, i)
            break
        end
    end

    return nothing
end

function caseIDsort(x::Node, y::Node)::Bool
    x.caseID < y.caseID
end

function DataFrames.nrow(dfr::DataFrameRow); return 1; end

struct IDOrdering <: Base.Order.Ordering; end

function Base.Order.lt(o::IDOrdering, a::Node, b::Node); return caseIDsort(a,b); end

struct TimeOrdering <: Base.Order.Ordering
end

function Base.Order.lt(o::TimeOrdering, a::Node, b::Node); return a.infection_time < b.infection_time; end

function Base.isless(x::Node, y::Node)::Bool; return  x.infection_time < y.infection_time; end

function Base.isless(x::event, y::event)::Bool; return  x.time < y.time; end

Base.deepcopy(m::branchModel) = branchModel([ deepcopy(getfield(m, k)) for k = 1:length(fieldnames(branchModel)) ]...)

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
        getOnsetToIsolDelay(model, nrow(detected_cases))
end

function initDataframe_isolation!(population_df::DataFrame, model::branchModel)
    #=
    Adds isolation parameters to our population dataframe.

    Currently detected cases are only clinical cases and they are Bernoulli random
    variables with probability model.p_test

    Acts in place on population_df
    =#

    new_model = deepcopy(model)
    new_model.max_cases = nrow(population_df)

    time_onset_delay = getOnsetDelay(new_model)
    time_isolated = ones(new_model.max_cases) .* Inf

    population_df.time_onset_delay = time_onset_delay
    population_df.time_isolated = time_isolated

    population_df.detected = convert.(Bool,zeros(new_model.max_cases))

    if model.p_test > 0
        clin_cases = filter(row -> row.sub_Clin_Case==false, population_df, view=true)
        clin_cases.detected .= rand(Bernoulli(model.p_test), nrow(clin_cases))
        detected_cases = filter(row -> row.detected==true, clin_cases, view=true)
        detected_cases.time_isolated .= getTimeIsolated(model, detected_cases)
    end
    return nothing
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

    active = BitArray(undef, model.max_cases) .* false
    # convert.(Bool,zeros(model.max_cases))
    active[1:model.state_totals[2]] .= true

    time_infected = Array{Float64}(undef, model.max_cases) .* 0.0
    time_infected[1:model.state_totals[2]] .= model.t_init * 1.0

    # init whether a given case is subClin or not.
    sub_Clin_Case = rand(model.max_cases) .< model.sub_clin_prop

    reproduction_number = getRi(model, sub_Clin_Case)

    # time_onset_delay = getOnsetDelay(model)
    # time_isolated = ones(model.max_cases) .* Inf

    population_df.parentID = parentID
    population_df.caseID = caseID
    population_df.active = active
    population_df.sub_Clin_Case = sub_Clin_Case
    population_df.reproduction_number = reproduction_number
    population_df.num_offspring = num_offspring
    population_df.time_infected = time_infected
    population_df.time_recovery = active .* model.recovery_time
    initDataframe_isolation!(population_df, model)

    # population_df.time_onset_delay = time_onset_delay
    # population_df.time_isolated = time_isolated
    #
    # population_df.detected = convert.(Bool,zeros(model.max_cases))
    # clin_cases = filter(row -> row.sub_Clin_Case==false, population_df, view=true)
    # clin_cases.detected .= rand(Bernoulli(model.p_test), length(clin_cases.active))
    #
    # if model.p_test > 0
    #     detected_cases = filter(row -> row.detected==true, clin_cases, view=true)
    #     detected_cases.time_isolated .= getTimeIsolated(model, detected_cases)
    # end


    return population_df
end

function simpleGraph_branch(population_df::DataFrame, max_num_nodes, single_root::Bool, root_node=1)
    #=
    Construct a infection tree, beginning with initial cases, containing
    min(max_num_nodes, num_cases in population_df)
    =#

    num_cases = nrow(population_df)
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

    # if timestep does not divide perfectly, stop before max_time
    # however we will run sims only where it does.

    # time span to solve on
    tspan = (model.t_init,model.t_max)

    # times to solve on
    t = [i for i=tspan[1]:time_step:tspan[end]]

    num_steps = convert(Int, floor((model.t_max - model.t_init) / time_step))

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

    state_totals_all = Array{Int64}(undef, times_length, length(model.state_totals)) .* 0
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

    return nothing
end

function initNewCase!(population_df::DataFrame, model::branchModel, infection_time::Union{Float64,Int64}, parentID::Int64, num_cases::Int64)
    #=
    Initialises the new case that just occured for the first react branch

    Works in place on the population dataframe
    =#

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

    return nothing
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

    return nothing
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
    population_df.sub_Clin_Case = sub_Clin_Case
    # population_df.num_offspring = num_offspring

    return population_df
end

function newDataframe_rows(model::branchModel, nrows::Int64, maxCaseID::Int64, no_offspring::Bool)
    new_model = deepcopy(model)
    new_model.max_cases = nrows # amount_above_limit
    new_model.state_totals = [0,0,0]
    new_population_df = initDataframe_thin(new_model)
    new_population_df.parentID .= 0
    new_population_df.caseID .+= maxCaseID
    if no_offspring
        new_population_df.num_offspring = Array{Int64}(undef, new_model.max_cases) .* 0
    else
        new_population_df.num_offspring = rand.(Poisson.(new_population_df.reproduction_number))
    end

    return new_population_df
end

function branchingProcess(population_df::DataFrame, model::branchModel, full_final_gen::Bool, noTimeDepend::Bool)
    #=
    full_final_gen specifies that the final generation (whichever one ends in the
    simulation going over or equally the max cases allowed), will be fully simulated
    rather than cut off at the max case limit. This will be useful for one of
    our thinning heuristics (want a full final gen).
    =#

    generation::Int64 = 1
    gen_range_dict = Dict{Int64, UnitRange}()
    num_cases::Int64 = model.state_totals[2]*1
    gen_range_dict[generation] = 1:num_cases

    population_df[gen_range_dict[generation], :generation_number] .= generation * 1

    population_df.num_offspring = rand.(Poisson.(population_df.reproduction_number))
    hitMaxCases = false

    # if noTimeDepend -> maxGen == model.t_max
    while !hitMaxCases && (!noTimeDepend || generation < model.t_max)

        # determine number of offspring
        num_off = @view population_df[gen_range_dict[generation], :num_offspring]

        # new generation
        total_off = sum(num_off)
        generation+=1
        gen_range_dict[generation] = num_cases+1:total_off+num_cases
        if total_off == 0
            population_df[num_cases+1:end, :num_offspring] .= 0
            gen_range_dict[generation] = num_cases+1:nrow(population_df)
            break
        end

        ########## Logic from discrete
        if gen_range_dict[generation][end] >= model.max_cases
            hitMaxCases = true
            if !full_final_gen
                gen_range_dict[generation] = gen_range_dict[generation][1]:model.max_cases

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

            elseif full_final_gen # technically no longer required (could just be else rather than elseif)
                # have a full final gen but hit max cases. population_df is only of
                # length max cases. Therefore, concatenate a new dataframe on top of it

                new_population_df = newDataframe_rows(model, abs(num_cases + total_off - model.max_cases), model.max_cases, true)
                population_df = vcat(population_df, new_population_df, cols=:orderequal)
                population_df[gen_range_dict[generation], :num_offspring] .= 0
            end
        end
        #######################

        population_df[gen_range_dict[generation], :generation_number] .= generation * 1
        parentIDs = makeVectorFromFrequency(num_off, collect(gen_range_dict[generation-1]))
        population_df[gen_range_dict[generation], :parentID] .= parentIDs

        num_cases += total_off
        # if hitMaxCases && !full_final_gen
        #     population_df[gen_range_dict[generation], :num_offspring] .= 0
        # end
    end

    return gen_range_dict, num_cases, population_df
end

function casesOverGenerations(model::branchModel, gen_range_dict::Dict{Int64,UnitRange})
    #=
    Generates the state totals for a simple SIR branching process where infected
    individuals in generation 1, infect R new people, who become generation 2,
    etc. I.e. all individuals in generation 1 recover after 1 generation.

    There is no 'real' time period infected, it is all unit length.

    stateTotals is a 3 col array [S,I,R], with rows corresponding to generation
    number
    =#

    max_generation = maximum(keys(gen_range_dict))
    generations = collect(1:max_generation)
    stateTotals = initStateTotals(model, max_generation)

    for row::Int64 in generations[2:end]
        stateTotals[row,1] = stateTotals[row-1,1] - length(gen_range_dict[row])
    end
    stateTotals[2:end,2] .= -diff(stateTotals[:,1])
    stateTotals[2:end,3] .= cumsum(stateTotals[1:end-1,2])

    return generations, stateTotals
end

function createDependencyTree(population_df::DataFrame, num_cases::Int64)
    #=
    Returns a directed dependency / infection tree of which nodes infected
    =#

    dependency_tree = SimpleDiGraph(num_cases)

    @simd for i in 1:num_cases
        @inbounds add_edge!(dependency_tree, population_df[i, :parentID], population_df[i, :caseID])
    end

    return dependency_tree
end

function createThinningTree(population_df::DataFrame, numSeedCases)
    #=
    Returns a infection tree of which nodes infected in a structure to allow fast
    thinning

    population_df should be unsorted by time, instead sorted by parentID
    i.e. sortperm(population_df.parentID) == collect(1:nrow(population_df))

    Is true if this is called after infection times are added to the dataframe,
    but before it is sorted within bpMain
    =#

    tree = Tree([Node(i, population_df[i, :time_infected], Ref(nothing), []) for i in 1:numSeedCases])

    childrenDeque = Deque{Node}()
    for node in tree.rootNodes
        if population_df[node.caseID, :num_offspring] != 0
            push!(childrenDeque, node)
        end
    end

    current_range = numSeedCases:numSeedCases

    # insert children
    while !isempty(childrenDeque)

        currentNode::Node = popfirst!(childrenDeque)
        current_offspring = copy(population_df[currentNode.caseID, :num_offspring])

        current_range = current_range[end]+1:current_range[end]+current_offspring

        currentNode.children = [Node(i, population_df[i, :time_infected], Ref(currentNode), []) for i in current_range]

        for node in currentNode.children
            if population_df[node.caseID, :num_offspring] != 0
                push!(childrenDeque, node)
            end
        end
    end

    return tree
end

function getTimeInfected_relativeArray(model::branchModel, num_cases)
    #=
    Returns an array of num_cases infection times
    =#

    infection_time_dist = Weibull(model.t_generation_shape, model.t_generation_scale)
    return rand(infection_time_dist, num_cases)
end

function getTimeInfected_relative!(population_df::DataFrame, model::branchModel, gen_range_dict, num_cases::Int64)
    #=
    Returns the time that an individual is infected by their parent. This is a
    time relative to the when the parent's infection occurred.

    Works in place on population_df, adding a new column
    =#

    infection_time_dist = Weibull(model.t_generation_shape, model.t_generation_scale)
    time_infected_rel = zeros(nrow(population_df))
    numSeedCases = length(gen_range_dict[1])

    time_infected_rel[numSeedCases+1:num_cases] .= rand(infection_time_dist, num_cases-numSeedCases)
    population_df.time_infected_rel = time_infected_rel

    return nothing
end

function getTimeInfected_abs!(population_df::DataFrame, model::branchModel,
    gen_range_dict::Dict, num_cases::Int64)
    #=
    Returns the actual time that an individual is infected by their parent

    Works in place on population_df, adding a new column
    =#

    numSeedCases = length(gen_range_dict[1])
    time_infected = zeros(nrow(population_df))

    for gen in 2:maximum(keys(gen_range_dict))
        infect_range = gen_range_dict[gen]

        time_infected[infect_range] .= population_df[infect_range, :time_infected_rel] .+ time_infected[population_df[infect_range, :parentID]]
    end

    population_df.time_infected = time_infected
    return nothing
end

function getTimeInfected_abs_tree!(population_df::DataFrame, model::branchModel,
    gen_range_dict::Dict, num_cases::Int, dependency_tree::SimpleDiGraph)
    #=
    Returns the actual time that an individual is infected by their parent

    Uses a dependency tree of infections to iterate through cases

    Works in place on population_df, adding a new column
    =#


    time_infected = zeros(model.max_cases)

    # iterate to the second to last generation
    for gen in 1:maximum(keys(gen_range_dict))-1
        for case in gen_range_dict[gen]

            children = outneighbors(dependency_tree, case)

            time_infected[children] .= population_df[children, :time_infected_rel] .+ time_infected[case]
        end
    end

    population_df.time_infected = time_infected

    return nothing
end

function getTimeRecovered!(population_df::DataFrame, model::branchModel)
    #=
    Returns the time that an infected individual recovers as a new column of
    the population_df dataframe.
    =#

    population_df.time_recovery = population_df.time_infected .+ model.recovery_time

    return nothing
end

function sortInfections(population_df::Union{SubDataFrame,DataFrame}, model::branchModel, viewBool::Bool=true)
    #=
    Returns a sorted view (if viewBool==true) of a population_df dataframe

    It is sorted by infection time
    =#

    return sort(population_df, [:time_infected], view=viewBool)
end

function filterInfections(sorted_df::Union{SubDataFrame,DataFrame}, model::branchModel, thinning::Bool, viewBool::Bool=false)
    #=
    Returns a filtered sorted_df dataframe, removing any cases that never got
    used.
    It is filtered by whether an infection is thinned
    =#
    if thinning
        return filter(row -> !row.thinned, sorted_df, view=viewBool)
    end
    return sorted_df
end

function mergeArrays_events(time_infected::Array{Float64,1}, time_recovery::Union{SubArray{Float64,1},Array{Float64,1}},
    t::Array{Float64,1}, stateTotals::Array{Int64,2})
    #=
    This is a modified version of: https://www.geeksforgeeks.org/merge-two-sorted-arrays/
    This is method 2. I.e. a merge sort on two sorted arrays

    Works in place on t and stateTotals arrays, ignoring the very first index.
    Creates the stateTotals array by initialising each row with the cumulative
    effect, and then performing a cumsum across these rows to get the overall
    change in totals over time.

    Length of combined infection and recovery arrays should be 1 less than length
    of t: n1+n2+1 == length(t)
    =#

    n1 = length(time_infected)
    n2 = length(time_recovery)

    t_ind = 2
    inf_ind = 1
    rec_ind = 1

    @assert n1+n2+1 == length(t)

    while inf_ind <= n1 && rec_ind <= n2
        if time_infected[inf_ind] < time_recovery[rec_ind]
            t[t_ind] = time_infected[inf_ind]
            stateTotals[t_ind, :] .= [-1,1,0]
            t_ind+=1; inf_ind+=1
        else
            t[t_ind] = time_recovery[rec_ind]
            stateTotals[t_ind, :] .= [0,-1,1]
            t_ind+=1; rec_ind+=1
        end

    end

    # Store remaining elements
    # of first array
    while inf_ind <= n1
        t[t_ind] = time_infected[inf_ind]
        stateTotals[t_ind, :] .= [-1,1,0]
        t_ind+=1; inf_ind+=1
    end

    # Not going to store additional recovery events due to thinning
    # Store remaining elements
    # of second array
    while rec_ind <= n2
        t[t_ind] = time_recovery[rec_ind]
        stateTotals[t_ind, :] .= [0,-1,1]
        t_ind+=1; rec_ind+=1
    end

    t = t[1:t_ind-1]
    stateTotals = stateTotals[1:t_ind-1,:]

    cumsum!(stateTotals, stateTotals, dims=1)

    return t, stateTotals
end

function findmaxgen_unthinned(sorted_df::Union{DataFrame,SubDataFrame}, numSeedCases::Int64, unthinned_caseID::Int64)
    #=
    Finds the first instance of the maximum generation that is unthinned in the sorted df between
    numSeedCases and thinned index (which is the index we have stopped thinning at)
    =#

    max_gen = 0
    caseID = 0

    for i in numSeedCases+1:unthinned_caseID
        if !sorted_df[i, :thinned] && sorted_df[i, :generation_number] > max_gen
            max_gen = sorted_df[i, :generation_number] * 1
            caseID = sorted_df[i, :caseID] * 1
        end
    end

    return max_gen, caseID
end

function findlastID_unthinned(sorted_df::Union{DataFrame,SubDataFrame}, numSeedCases::Int64, unthinned_caseID::Int64)
    #=
    Finds the last caseID that is unthinned in the sorted df between
    numSeedCases and thinned index (which is the index we have stopped thinning at)
    =#

    for i in unthinned_caseID:-1:numSeedCases+1
        if !sorted_df[i, :thinned]
            return sorted_df[i, :caseID]
        end
    end
    return 0
end

function dataframeClean!(sorted_df::DataFrame, range_to_clean::UnitRange)
    #=
    Sets all relevant columns of a dataframe to 0 (0.0 if Float and false if Bool)
    =#

    sorted_df[range_to_clean, :num_offspring] .= 0
    sorted_df[range_to_clean, :time_infected] .= 0.0
    sorted_df[range_to_clean, :time_infected_rel] .= 0.0
    sorted_df[range_to_clean, :time_recovery] .= 0.0
    sorted_df[range_to_clean, :parentID] .= 0
    sorted_df[range_to_clean, :generation_number] .= 0
    sorted_df[range_to_clean, :offspring_simmed] .= false
    return nothing
end

function simChildren_ThinTree!(population_df::DataFrame, model::branchModel,
    currentNode::Node, new_model::branchModel, sSaturation::Float64, isolationDepend::Bool, thinnedToReuse::Deque{Node})
    #=
    Simulate the children of a given node.

    May add support for multiple generations simulated
    =#

    expOff = population_df[currentNode.caseID, :reproduction_number] * sSaturation
    numOff = rand(Poisson(expOff))
    population_df[currentNode.caseID, :num_offspring] = numOff * 1

    newGenNumber = population_df[currentNode.caseID, :generation_number]+1
    current_time_infected = population_df[currentNode.caseID, :time_infected]

    while numOff != 0 && !isempty(thinnedToReuse)
        newChild = pop!(thinnedToReuse)

        population_df[newChild.caseID, :parentID] = currentNode.caseID
        population_df[newChild.caseID, :generation_number] = newGenNumber
        population_df[newChild.caseID, :time_infected_rel] = getTimeInfected_relativeArray(model, 1)[1]
        population_df[newChild.caseID, :time_infected] = population_df[newChild.caseID, :time_infected_rel] + current_time_infected
        population_df[newChild.caseID, :time_recovery] = population_df[newChild.caseID, :time_infected] + model.recovery_time

        newChild.infection_time = population_df[newChild.caseID, :time_infected]
        newChild.parent = Ref(currentNode)

        # update gen number, time_infected and time_isolated (if applicable)
        if isolationDepend

            if population_df[newChild.caseID, :detected]
                population_df[newChild.caseID, :time_isolated] = getTimeIsolated(model, population_df[newChild.caseID, :])[1]
            end

        end

        push!(currentNode.children, newChild)

        numOff -= 1
    end

    if numOff != 0
        # add rows to df

        new_model.max_cases = numOff
        new_model.state_totals[2] = 0
        new_rows = initDataframe_thin(new_model)

        new_rows.parentID .= currentNode.caseID
        new_rows.caseID .+= nrow(population_df)
        new_rows.generation_number .= newGenNumber
        new_rows.num_offspring = Array{Int64}(undef, numOff) .* 0
        new_rows.time_infected_rel = getTimeInfected_relativeArray(model, numOff)
        new_rows.time_infected = new_rows.time_infected_rel .+ current_time_infected
        getTimeRecovered!(new_rows, model)
        new_rows.sSaturation_upper = ones(numOff)
        new_rows.thinned = BitArray(undef, numOff) .* false
        new_rows.thinned .= true
        new_rows.offspring_simmed = BitArray(undef, numOff) .* false
        new_rows.reused = BitArray(undef, numOff) .* false

        if isolationDepend
            initDataframe_isolation!(new_rows, new_model)
        end

        append!(population_df, new_rows, cols=:setequal)
        # population_df = vcat(population_df, new_rows)

        current_range = nrow(population_df)-numOff+1:nrow(population_df)

        push!(currentNode.children, [Node(i, population_df[i, :time_infected], Ref(currentNode), []) for i in current_range]...)
    end

    return nothing
end

function bp_ThinTree!(population_df::DataFrame,
    model::branchModel, gen_range_dict, num_cases::Int64, saturationDepend::Bool,
    isolationDepend::Bool, thinningTree::Tree)
    #=
    Thins events from the population_df dataframe, by setting a boolean column
    to true for that row.

    Performs this by iterating through a directed tree in order of infection
    time and only considering a node's children if it has not been thinned.

    Thins based on saturation factors and isolation of cases. Could also add
    support for alert level changes (as in Mike et. al.)

    Stopping conditions are:
        if search (no children left) becomes empty
        if hit max cases / population size
        if hit t_max
    =#

    popsortvector = Array{Int64}(undef, model.max_cases) .* 0
    popsortvector[gen_range_dict[1]] = population_df[gen_range_dict[1], :caseID]
    popsortindex = gen_range_dict[1][end]

    search = BinaryHeap{Node, TimeOrdering}()
    thinnedToReuse = Deque{Node}()
    reused = BitArray(undef, nrow(population_df)) .* false
    population_df.reused = reused

    # push all children of rootNodes into heap. They will be sorted by infection time
    for rootNode in thinningTree.rootNodes
        for child in rootNode.children
            push!(search, child)
        end
    end

    # saturation factors
    sSaturation = (model.state_totals[1]/model.population_size)
    sSaturation_upper = ones(nrow(population_df))
    population_df.sSaturation_upper = sSaturation_upper
    increment = 1.0/model.population_size

    # thinned column
    thinned = BitArray(undef, nrow(population_df)) .* false
    thinned[gen_range_dict[1][end]+1:end] .= true
    population_df.thinned = thinned

    # offspring simmed column (only haven't simmed offspring from final generation)
    offspring_simmed = BitArray(undef, nrow(population_df)) .* false
    offspring_simmed .= true
    offspring_simmed[gen_range_dict[maximum(keys(gen_range_dict))]] .= false
    population_df.offspring_simmed = offspring_simmed

    total_thinned = 0
    new_model = deepcopy(model) # for creating new dataframes

    while !isempty(search)
        currentNode = pop!(search)

        if saturationDepend && rand() > (sSaturation/population_df[currentNode.parent[].caseID::Int64, :sSaturation_upper])
            # you are thinned
            deleteChildNode!(currentNode)
            population_df[currentNode.parent[].caseID::Int64, :num_offspring]-=1

            currentNode.parent = Ref(nothing)
            push!(thinnedToReuse, currentNode)
            population_df[currentNode.caseID, :reused] = true

        elseif isolationDepend && population_df[currentNode.parent[].caseID, :detected] &&
            population_df[currentNode.parent[].caseID,:time_isolated] < currentNode.infection_time &&
            model.isolation_infectivity < rand()

            deleteChildNode!(currentNode)
            population_df[currentNode.parent[].caseID::Int64, :num_offspring]-=1

            currentNode.parent = Ref(nothing)
            push!(thinnedToReuse, currentNode)
            population_df[currentNode.caseID, :reused] = true

        else

            popsortindex+=1
            popsortvector[popsortindex] = currentNode.caseID
            population_df[currentNode.caseID, :thinned] = false

            # we have simmed exactly, a total of model.max_cases (stopping condition)
            if popsortindex == model.max_cases
                break
            end

            sSaturation-=increment

            # offspring not simmed
            if !population_df[currentNode.caseID, :offspring_simmed]
                population_df[currentNode.caseID, :sSaturation_upper] = sSaturation*1
                # Simulate offspring for a few generations (one?)
                # simulate at sSaturation_upper/sSaturation
                simChildren_ThinTree!(population_df, model, currentNode, new_model, sSaturation, isolationDepend, thinnedToReuse)
            end

            # reused an event. Need to rescale t_infect and t_isolated for their children. Both in df and in the Node tree
            if population_df[currentNode.caseID, :reused]
                if !isempty(currentNode.children)

                    child1 = currentNode.children[1]

                    old_parent_t_infect = population_df[child1.caseID, :time_infected] - population_df[child1.caseID, :time_infected_rel]
                    t_diff = currentNode.infection_time - old_parent_t_infect
                    newGen = population_df[currentNode.caseID, :generation_number] + 1

                    for child in currentNode.children
                        child.infection_time += t_diff
                        population_df[child.caseID, :generation_number] = newGen
                        population_df[child.caseID, :time_infected] += t_diff
                        population_df[child.caseID, :time_recovery] += t_diff
                        population_df[child.caseID, :reused] = true

                        if isolationDepend
                            population_df[child.caseID, :time_isolated] += t_diff
                        end
                    end
                end
            end

            # add children to search
            for child in currentNode.children
                push!(search, child)
            end
        end
    end

    return population_df[popsortvector[1:popsortindex], :]
end

function bp_SingleThin!(population_df::DataFrame,
    model::branchModel, gen_range_dict, num_cases::Int64, saturationDepend::Bool,
    isolationDepend::Bool)
    #=
    Thins events from the population_df dataframe, by setting a boolean column
    to true for that row

    One single thinning iteration.

    Continue thinning until sSaturation <= lower bound (0.1 atm)
    =#

    sorted_df = sortInfections(population_df, model, false)

    sSaturation = (model.state_totals[1]/model.population_size)
    caseSaturation = zeros(num_cases) # corresponds to the sorted indexes
    caseSaturation[gen_range_dict[1]] .= sSaturation*1
    increment = 1.0/model.population_size

    caseIDs_to_index = sparsevec(sorted_df.caseID, collect(1:nrow(sorted_df)))
    unthinned_caseID = 0

    thinned = BitArray(undef, nrow(sorted_df)) .* false
    thinned[gen_range_dict[1][end]+1:end] .= true
    sorted_df.thinned = thinned

    numSeedCases = length(gen_range_dict[1])

    total_thinned = 0
    parentIndex = 0

    # thinning of child cases
    for i::Int64 in numSeedCases+1:nrow(sorted_df)
        parentIndex::Int64 = caseIDs_to_index[sorted_df[i, :parentID]]*1
        # saturationThin
        if sorted_df[parentIndex, :thinned] || (saturationDepend && rand()>sSaturation)#/sSaturation_upper
            total_thinned+=1
            sorted_df[parentIndex, :num_offspring]-=1

            caseSaturation[i] = caseSaturation[i-1]*1

            # isolThin
        elseif isolationDepend && sorted_df[parentIndex, :detected] &&
            sorted_df[parentIndex, :time_isolated] < sorted_df[i, :time_infected] &&
            model.isolation_infectivity < rand()

            total_thinned+=1
            sorted_df[parentIndex, :num_offspring]-=1

            caseSaturation[i] = caseSaturation[i-1]*1
        else
            caseSaturation[i] = sSaturation*1
            sSaturation-=increment
            sorted_df[i, :thinned] = false
        end

        if sSaturation <= 0.1
            # println("Hit saturation lower bound")
            break
        end

        # if sorted_df[i, :time_infected] > model.t_max
        #     break
        # end

    end
    # println("Total thinned is $total_thinned")

    return filterInfections(sorted_df, model, true)
end

function bp_thinnedHereAndNow!(population_df::DataFrame,
    model::branchModel, gen_range_dict, num_cases::Int64, saturationDepend::Bool,
    isolationDepend::Bool, prob_accept::Float64)
    #=
    Thins events from the population_df dataframe, by setting a boolean column
    to true for that row

    Continue thinning until sSaturation / sSaturation_upper <= prob_accept
    =#

    sorted_df = sortInfections(population_df, model)

    sSaturation = (model.state_totals[1]/model.population_size)
    sSaturation_upper = 1.0 #(model.state_totals[1]/model.population_size)
    caseSaturation = zeros(num_cases) # corresponds to the sorted indexes
    caseSaturation[gen_range_dict[1]] .= sSaturation*1

    caseIDs_to_index = sparsevec(sorted_df.caseID, collect(1:nrow(sorted_df)))
    unthinned_caseID = 0

    increment = 1.0/model.population_size
    thinned = BitArray(undef, num_cases) .* false
    population_df.thinned = thinned

    offspring_simmed = BitArray(undef, num_cases) .* false
    offspring_simmed .= true
    offspring_simmed[gen_range_dict[maximum(keys(gen_range_dict))]] .= false
    population_df.offspring_simmed = offspring_simmed
    numSeedCases = length(gen_range_dict[1])

    total_thinned = 0
    it_count = 1

    while it_count < 10000
        it_count+=1
        # thinning of child cases
        # maxGen = maximum(keys(gen_range_dict))

        for i in numSeedCases+1:nrow(sorted_df)
            parentIndex = caseIDs_to_index[sorted_df[i, :parentID]]

            # saturationThin
            if saturationDepend && (sorted_df[parentIndex, :thinned] || rand()>sSaturation/sSaturation_upper)
                # thinned[sorted_df[i, :caseID]] = true
                sorted_df[i, :thinned] = true
                total_thinned+=1
                sorted_df[parentIndex, :num_offspring]-=1
                # population_df[parentID, :num_offspring]-=1

                caseSaturation[i] = caseSaturation[i-1]*1

                # isolThin
            elseif isolationDepend && sorted_df[parentIndex, :detected] &&
                sorted_df[parentIndex, :time_isolated] < sorted_df[i, :time_infected] &&
                model.isolation_infectivity < rand()

                sorted_df[i, :thinned] = true
                total_thinned+=1
                sorted_df[parentIndex, :num_offspring]-=1
                # population_df[parentID, :num_offspring]-=1

                caseSaturation[i] = caseSaturation[i-1]*1

            else
                caseSaturation[i] = sSaturation*1
                sSaturation-=increment
                # parentSaturation[sorted_df[i, :caseID]] = sSaturation*1

                if !sorted_df[i, :offspring_simmed]
                    unthinned_caseID = sorted_df[i, :caseID]
                    # println("Offspring not simmed")
                    break
                end
            end

            if sSaturation <= 0.1
                println("Hit saturation lower bound")
                break
            end

            if sSaturation/sSaturation_upper <= prob_accept
                println("Hit p_accept")
                max_unthinned_gen, caseID = findmaxgen_unthinned(sorted_df, numSeedCases, i)

                if max_unthinned_gen == maximum(keys(gen_range_dict))
                    unthinned_caseID = caseID
                else
                    unthinned_caseID = findlastID_unthinned(sorted_df, numSeedCases, i)
                end
                break
            end
        end

        if unthinned_caseID == 0
            max_unthinned_gen, caseID = findmaxgen_unthinned(sorted_df, numSeedCases, nrow(sorted_df))
            println("New max gen is $max_unthinned_gen")
            if max_unthinned_gen == maximum(keys(gen_range_dict))
                unthinned_caseID = caseID
            else
                unthinned_caseID = findlastID_unthinned(sorted_df, numSeedCases, nrow(sorted_df))
            end
        end

        # we will break if it is still 0 after the above for loop
        if unthinned_caseID == 0
            println("All possible events thinned")
            break
        end

        # Heuristic one - we throw out all cases after unthinned_caseID
        # then simulate new cases from here first react / next react style
        # easy conceptually. Hard code structure-wise
        if true #H1

            # we clean all events after unthinned_caseID
            for i in caseIDs_to_index[unthinned_caseID]+1:nrow(sorted_df)
                parentIndex = caseIDs_to_index[sorted_df[i, :parentID]]
                # parentID = sorted_df[i, :parentID]
                if sorted_df[parentIndex, :thinned] #!thinned[parentID]
                    sorted_df[parentIndex, :num_offspring]-=1
                    # population_df[parentID, :num_offspring]-=1 # need to throw out offspring that occur after unthinned_caseID
                end
            end

            # reset thin IDS for everyone after current case
            # thinned[sorted_df[caseIDs_to_index[unthinned_caseID]+1:end, :caseID]] .= false
            sorted_df[caseIDs_to_index[unthinned_caseID]+1:end, :thinned] .= false
            # population_df.thinned = thinned

            # new sorted_df
            sorted_df = filterInfections(sorted_df, model, true)

            sorted_df_len = nrow(sorted_df)
            caseIDs_to_index = sparsevec(sorted_df.caseID, collect(1:sorted_df_len))

            # throw out/clean all values after unthinned_caseID
            range_to_clean = caseIDs_to_index[unthinned_caseID]+1:sorted_df_len
            dataframeClean!(sorted_df, range_to_clean)

            # local time is infection time of unthinned_caseID
            t_current=0.0
            try
                t_current = sorted_df[caseIDs_to_index[unthinned_caseID], :time_infected]
            catch BoundsError
                println(unthinned_caseID)
                println(caseIDs_to_index[unthinned_caseID])
                println(minimum(caseIDs_to_index))
                println(sum(sorted_df.caseID .== unthinned_caseID))
                @error "BoundsError"
            end
            sorted_df.active = sorted_df.time_recovery .> t_current
            active_range = findfirst(sorted_df.active):findlast(sorted_df.active)

            # println(active_range)

            # sometimes error produced from here - if sSaturation_upper is neg then
            # will break poisson expOff
            # = (|S|) / N. i.e. |S| = N - |I| - |R| i.e. num still to infect
            sSaturation_upper = (model.population_size-caseIDs_to_index[unthinned_caseID]) / model.population_size


            # perform a first react step but let all reactions occur.
            # firstReact_step!()

            #-------------------------------
            # filter df on active infections
            infection_time_dist = Weibull(model.t_generation_shape, model.t_generation_scale)
            active_df = filter(row -> row.active, sorted_df, view=true)

            active_time_left = active_df.time_recovery .- t_current
            active_time_spent = round.(model.recovery_time .- active_time_left, digits=9)

            # Draw number of reactions for each active infected individual
            # expected number of reactions in time remaining, given time distribution
            expOff = (sSaturation_upper) .*
                active_df.reproduction_number .* ccdf.(infection_time_dist, t_current.-active_df.time_infected)

            num_off = rand.(Poisson.(expOff))

            if !isa(num_off, Array)
                num_off = [num_off]
            end

            # add all these new events. Re-sort them afterwards
            total_off = sum(num_off)

            if total_off == 0
                # return filter(row->row.time_recovery > 0.1, sorted_df, view=false)
                println("No new cases")
                break
            end

            active_df.num_offspring .= num_off

            parentIDs = makeVectorFromFrequency(num_off, active_df.caseID)
            infection_time = zeros(total_off)
            currentRange = 0:0

            for i in 1:length(num_off)
                if num_off[i] != 0
                    currentRange = currentRange[end]+1:currentRange[end]+num_off[i]
                    cdf_at_time = cdf(infection_time_dist, active_time_spent[i])

                    ###### multiple samples for each offspring
                    log_transformed_rand = log.(rand(num_off[i]) .* (1-cdf_at_time) .+ cdf_at_time)

                    timeToReact = invlogcdf.(infection_time_dist, log_transformed_rand) .- active_time_spent[i]

                    infection_time[currentRange] .= timeToReact .+ t_current
                end
            end

            perm_vector = sortperm(infection_time)

            rem_rows = sorted_df_len - active_range[end]
            if rem_rows < total_off
                new_sorted_df = newDataframe_rows(model, abs(rem_rows - total_off), maximum(sorted_df.caseID), false)
                new_sorted_df.thinned = BitArray(undef, nrow(new_sorted_df)) .* false
                new_sorted_df.time_infected = Array{Float64}(undef, nrow(new_sorted_df)) .* 0
                new_sorted_df.time_infected_rel = Array{Float64}(undef, nrow(new_sorted_df)) .* 0
                new_sorted_df.time_recovery = Array{Float64}(undef, nrow(new_sorted_df)) .* 0
                new_sorted_df.active = BitArray(undef, nrow(new_sorted_df)) .* false
                new_sorted_df.offspring_simmed = BitArray(undef, nrow(new_sorted_df)) .* false

                sorted_df = vcat(sorted_df, new_sorted_df, cols=:setequal)

                sorted_df_len = nrow(sorted_df)
                caseIDs_to_index = sparsevec(sorted_df.caseID, collect(1:sorted_df_len))
            end

            offspring_range = active_range[end]+1:active_range[end]+total_off
            sorted_df[offspring_range, :parentID] .= parentIDs[perm_vector]
            sorted_df[offspring_range, :time_infected] .= infection_time[perm_vector]
            sorted_df[offspring_range, :time_recovery] .= sorted_df[offspring_range, :time_infected] .+ 30
            sorted_df[offspring_range, :generation_number] .=  sorted_df[caseIDs_to_index[sorted_df[offspring_range, :parentID]], :generation_number] .+1
            active_df.offspring_simmed .= true

            # will need to repeat this for the next react step (if it occurs)
            newMaxGen = maximum(sorted_df[offspring_range, :generation_number])
            # println("New max gen is $newMaxGen")

            if newMaxGen > maximum(keys(gen_range_dict))
                gen_range_dict[newMaxGen] = 1:1
                println("New max gen")
            end

            # need to re-sort overall sorted_df

            #then perform next react steps from here
            # at least 3 steps I think. Maybe 4-5. Probs no more than 8



            #--------------------------------
            # return filter(row->row.time_recovery > 0.1, sorted_df, view=false)


            sorted_df = sortInfections(sorted_df, model)
            sorted_df = filter(row->row.time_recovery > 0.1, sorted_df, view=false)
            numSeedCases = caseIDs_to_index[unthinned_caseID] # the position in sorted_df to thin from
            caseSaturation = zeros(nrow(sorted_df))
            caseSaturation[1:numSeedCases] .= sSaturation_upper * 1
            sSaturation = sSaturation_upper * 1
            unthinned_caseID = 0
            caseIDs_to_index = sparsevec(sorted_df.caseID, collect(1:nrow(sorted_df)))

            # thinned = sparsevec(sorted_df.caseID, sorted_df.thinned)

        end

        if rem(it_count, 100) == 0
            println("Iteration number $it_count")
        end
    end

    println("Total thinned is $total_thinned")

    return filter(row->row.time_recovery > 0.1, sorted_df, view=false)
end

function casesOverTime(sorted_df::Union{DataFrame,SubDataFrame}, model::branchModel, gen_range_dict::Dict)
    #=
    Generates the state totals for a complex SIR branching process where an infected
    individual, i, infects R_i people on average. May or may not include thinning
    for time dependent processes which have here and now uncertainty (this occurs
    before this function). This function includes thinning for wait and see
    uncertainty - i.e. if we hit a case limit and an alert level changes, which
    can only be determined at this point in the process.

    An infected person recovers at time_recovery - this will tend to be a fixed
    period after they have been infected.

    stateTotals is a 3 col array [S,I,R], with rows corresponding to time
    =#

    num_cases = nrow(sorted_df)
    numSeedCases = length(gen_range_dict[1])

    # twice as long array to allow for recovery events too
    t = zeros(num_cases*2 - numSeedCases+1)
    stateTotals = initStateTotals(model, length(t))

    t[1] = model.t_init
    stateTotals[1,:] = model.state_totals

    t, stateTotals = mergeArrays_events(sorted_df.time_infected[numSeedCases+1:end], sorted_df.time_recovery, t, stateTotals)

    if t[end] >= model.recovery_time
        # clean duplicate values of t which occur on the first recovery time
        firstDupe = findfirst(x->x==model.recovery_time,t)
        lastDupe = findlast(x->x==model.recovery_time,t)

        t = vcat(t[1:firstDupe-1], t[lastDupe:end])
        stateTotals = vcat(stateTotals[1:firstDupe-1, :], stateTotals[lastDupe:end, :])
    end

    return t, stateTotals
end

function bpMain!(population_df::DataFrame, model::branchModel, noTimeDepend::Bool,
    thinMethod::ThinFunction=ThinFunction(ThinTree()),
    full_final_gen::Bool=false, sDepend::Bool=true, isolDepend::Bool=false)
    #=
    No longer works in place on population_df. Instead has to pass a copy round
    of population_df due to the actions of function: branchingProcess, which
    uses a vcat on the dataframe if full_final_gen == true.
    =#

    @assert model.max_cases > model.state_totals[2] "The number of allowable cases must be greater than the number of seed cases"

    if thinMethod.name == ThinTree() || thinMethod.name == ThinSingle()
        full_final_gen = true
    end

    gen_range_dict, num_cases, population_df = branchingProcess(population_df, model, full_final_gen, noTimeDepend)

    if noTimeDepend
        generations, state_totals_all = casesOverGenerations(model, gen_range_dict)
        return generations, state_totals_all, population_df
    end

    getTimeInfected_relative!(population_df, model, gen_range_dict, num_cases)
    getTimeInfected_abs!(population_df, model, gen_range_dict, num_cases)
    # dependency_tree = createDependencyTree(population_df, num_cases)
    # getTimeInfected_abs_tree!(population_df, model, gen_range_dict, num_cases, dependency_tree)

    getTimeRecovered!(population_df, model)

    sorted_df = DataFrame()

    if thinMethod.name != ThinNone && (sDepend || isolDepend)

        # do thinning
        # sDepend=true
        # check if we are using test and isolation thinning
        # isolDepend=false

        if isolDepend
            initDataframe_isolation!(population_df, model)
        end

        # thinType = "firstReactThin"
        thinType = "thinningTree"
        # thinType = "singleThin"

        @assert thinType in ["thinningTree", "firstReactThin", "singleThin"] "Thinning function $thinType does not exist"


        prob_accept = 0.9

        if thinMethod.name == ThinTree()
            # need full_final_gen = true for this kind of thinning
            tree = createThinningTree(population_df, length(gen_range_dict[1]))
            sorted_df = bp_ThinTree!(population_df, model, gen_range_dict, num_cases, sDepend, isolDepend, tree)
        end
        if thinMethod.name == ThinFirst()
            sorted_df = bp_thinnedHereAndNow!(population_df, model, gen_range_dict, num_cases, sDepend, isolDepend, prob_accept)
        end
        if thinMethod.name == ThinSingle()
            sorted_df = bp_SingleThin!(population_df, model, gen_range_dict, num_cases, sDepend, isolDepend)
        end
    else
        sorted_df = sortInfections(population_df, model)
    end

    t, state_totals_all = casesOverTime(sorted_df, model, gen_range_dict)

    return t, state_totals_all, sorted_df
end

function discrete_branch!(population_df::DataFrame, model::branchModel, time_step::Number)
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

function firstReact_branch!(population_df::DataFrame, model::branchModel)

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

function nextReact_branch_trackedHeap!(population_df::DataFrame, model::branchModel)

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

function nextReact_branch!(population_df::DataFrame, model::branchModel)

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


    while t[end] < model.t_max && model.state_totals[2] != 0 &&
        num_cases < model.max_cases && !isempty(tau_heap)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = firstReact_branch!(population_df, model)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = firstReact_branch!(population_df, model)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = firstReact_branch!(population_df, model)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = firstReact_branch!(population_df, model)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = firstReact_branch!(population_df, model)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = nextReact_branch!(population_df, model)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = nextReact_branch!(population_df, model)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = nextReact_branch!(population_df, model)

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
            t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
            t, state_totals_all, num_cases = nextReact_branch!(population_df, model)

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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            # Simple branch, random infection times, s saturation thinning
            population_df = initDataframe_thin(model);

            t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinTree()), true, true, false)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, model)

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

        title = "Next React vs Simple Branch with Saturation Thinning"
        outputFileName = "./verifiedBranch/NextvsSimpleBP"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true, false)
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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            # Simple branch, random infection times, s saturation and isolation thinning
            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            model.p_test = 1.0
            model.sub_clin_prop = 0
            model.stochasticIsol = false
            # model.t_onset_shape = 5.8
            model.t_onset_to_isol = 0

            population_df = initDataframe_thin(model);
            t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinTree()), false, true, true)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            model.p_test = 1.0
            model.sub_clin_prop = 0
            model.stochasticIsol = false
            # model.t_onset_shape = 5.8
            model.t_onset_to_isol = 0

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, model)

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

        title = "Next React vs Simple BP with Isolation and S Saturation"
        outputFileName = "./verifiedBranch/NextvsSimpleBPIsolation"
        branchVerifyPlot(Smean, Imean, Rmean, discreteSIR_mean, times, title, outputFileName, false, true, true, false)
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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 1*10^3, [5*10^3-10,10,0]);

            # Simple branch, random infection times, s saturation thinning
            population_df = initDataframe_thin(model);

            t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinSingle()), true, true, false)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, model)

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==model.recovery_time,t)
            lastDupe = findlast(x->x==model.recovery_time,t)

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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            # Simple branch, random infection times, s saturation and isolation thinning
            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 1*10^3, [5*10^3-10,10,0]);
            model.p_test = 1.0
            model.sub_clin_prop = 0
            model.stochasticIsol = false
            # model.t_onset_shape = 5.8
            model.t_onset_to_isol = 0

            population_df = initDataframe_thin(model);
            t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinSingle()), false, true, true)

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

            model = init_model_pars(tspan[1], tspan[end], 5*10^3, 5*10^3, [5*10^3-10,10,0]);
            model.p_test = 1.0
            model.sub_clin_prop = 0
            model.stochasticIsol = false
            # model.t_onset_shape = 5.8
            model.t_onset_to_isol = 0

            population_df = initDataframe(model);
            t, state_totals_all, num_cases = nextReact_branch!(population_df, model)

            # clean duplicate values of t which occur on the first recovery time
            firstDupe = findfirst(x->x==model.recovery_time,t)
            lastDupe = findlast(x->x==model.recovery_time,t)

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

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-10,10,0]);
            # model.stochasticRi = false
            # model.sub_clin_prop = 0.0
            # model.reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(model);

            t, state_totals_all, population_df = bpMain!(population_df, model, true, ThinFunction(ThinNone()), true, false, false)

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

        # Iovertime = model.state_totals[2] .* (((model.reproduction_number*(1-model.sub_clin_prop) +
        #     model.reproduction_number*model.sub_clin_scaling*model.sub_clin_prop)*3.3/3).^times)

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

        title = "Exponential vs Simple Branch"
        outputFileName = "./verifiedBranch/ExponentialvsSimpleBP"
        branch2wayVerifyPlot(meanCumI, Iovertime, times, title, outputFileName, Dots(), true, true)
        # branch2wayVerifyPlot(Itmean, Iovertime, times, title, outputFileName, true, true)
    end

    println("Test #16: Epidemic curves (Simple BP Infections vs Exponential, distributed times)")
    if 16 in testRange
        println("Beginning simulation of Simple BP Case")

        # time span to sim on
        tspan = (0.0,30.0)
        time_step = 1

        # times to sim on
        times = [i for i=tspan[1]:time_step:tspan[end]]

        numSims = convert(Int, round(100 / numSimsScaling))

        StStep, ItStep, RtStep = initSIRArrays(tspan, time_step, numSims)

        sumOffspring = 0
        total_cases = 0

        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            model = init_model_pars(tspan[1], tspan[end], 5*10^4, 5*10^4, [5*10^4-20,20,0]);
            model.stochasticRi = false
            model.sub_clin_prop = 0.0
            # model.reproduction_number = 1.0

            # Simple branch, no infection times
            population_df = initDataframe_thin(model);

            t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinNone()), true, false, false)

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
        model.sub_clin_prop = 0.0
        # model.reproduction_number = 1.0

        # Iovertime = model.state_totals[2] .* (((model.reproduction_number*(1-model.sub_clin_prop) +
        #     model.reproduction_number*model.sub_clin_scaling*model.sub_clin_prop)*3.3/3).^times)

        meanInfectTime = mean(Weibull(model.t_generation_shape, model.t_generation_scale))
        normTimes = (times .- meanInfectTime) ./ (meanInfectTime * 1)

        Iovertime = model.state_totals[2] .* ((model.reproduction_number*(1-model.sub_clin_prop) +
            model.reproduction_number*model.sub_clin_scaling*model.sub_clin_prop).^(normTimes))

        Iovertime = cumsum(Iovertime)

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
        println("Finished Simulation in $time seconds")

        # Itmean= mean(It, dims = 2)

        endminus = 2
        meanCumI = meanCumI[1:end-endminus]

        # Itmean = Itmean[1:end-endminus]
        Iovertime = Iovertime[1:end-endminus]
        times = times[1:end-endminus]

        misfitI = sum(abs.(meanCumI - Iovertime))/length(meanCumI)
        println("Mean Abs Error I = $misfitI")

        title = "Exponential vs Simple Branch, Distributed Times"
        outputFileName = "./verifiedBranch/ExponentialvsSimpleBPTimes.png"
        branch2wayVerifyPlot(meanCumI, Iovertime, times, title, outputFileName, Lines(), true, true)
        # branch2wayVerifyPlot(Itmean, Iovertime, times, title, outputFileName, true, true)
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
        t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

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
    @time t, state_totals_all, num_cases = discrete_branch!(population_df, model, time_step)

    # first
    model = init_model_pars(0, 20, 5*10^7, 100, [5*10^7-10,10,0])
    population_df = initDataframe(model)
    @time t, state_totals_all, num_cases = firstReact_branch!(population_df, model)

    # next
    model = init_model_pars(0, 100, 5*10^3, 5*10^3, [5*10^3-10,10,0])
    population_df = initDataframe(model);
    @time t, state_totals_all, num_cases= nextReact_branch!(population_df, model)

    # Simple branch, random infection times, isolation & saturation thinning
    model = init_model_pars(0, 200, 5*10^3, 5*10^3, [5*10^3-10,10,0]);
    model.p_test = 1.0
    model.sub_clin_prop = 0
    model.stochasticIsol = false
    model.t_onset_to_isol = 0
    population_df = initDataframe_thin(model);
    @time t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinTree()), true, true, true)

    # Simple branch, random infection times, isolation & saturation thinning
    model = init_model_pars(0, 200, 5*10^3, 5*10^3, [5*10^3-10,10,0]);
    model.p_test = 1.0
    model.sub_clin_prop = 0
    model.stochasticIsol = false
    model.t_onset_to_isol = 0
    population_df = initDataframe_thin(model);
    @time t, state_totals_all, population_df = bpMain!(population_df, model, false, ThinFunction(ThinSingle()), true, true, true)
end

compilationInit()

verifySolutions(1, [15, 16])

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
