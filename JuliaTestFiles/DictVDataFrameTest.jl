#=
Test the relative performance of storing a population's data in a nested dictionary
and a DataFrame.

Demonstrably better performance found using a DataFrame
=#

using DataFrames

function test2()
    # test DataFrame
    n = 10000000

    networkVertex_df1 = DataFrame()
    networkVertex_df2 = DataFrame(state=String[], infectedNeighbours=Int64[], connectivity=Int64[],
        state1=Char[])

    # create df from columns | SIGNIFICANTLY FASTER (About 28x for n = 10000000)
    if true
        # initialising long string array
        state = Array{String}(undef, n)
        state .= "S"

        # try char instead of string
        state1 = Array{Char}(undef, n)
        state1 .= 'S'

        # faster than converting from zeros
        connectivity = Array{Int64}(undef, n)
        connectivity .= 0

        stateIndex = Array{Int64}(undef, n)
        stateIndex .= 1

        infectedNeighbours = Array{Int64}(undef, n)
        infectedNeighbours .= 0

        # infectedNeighbours = convert.(Int64, zeros(n))

        networkVertex_df1.connectivity = connectivity
        networkVertex_df1.state = state
        networkVertex_df1.state1 = state1
        networkVertex_df1.stateIndex = stateIndex
        networkVertex_df1.infectedNeighbours = infectedNeighbours
    end

    # create df row by row
    if false
        for i::Int64 in 1:n

            push!(networkVertex_df2, ("S", 0, 0, 'S'))

        end
    end

    # --------------------------------------------------------------------------
    states = ["I", "R"]
    states1 = ['I', 'R']
    stateIndexes = [2, 3]

    for i::Int64 in 1:n
        @inbounds networkVertex_df1[i, :state] = states[i%2 + 1]
        @inbounds networkVertex_df1[i, :state1] = states1[i%2 + 1]
        @inbounds networkVertex_df1[i, :stateIndex] = stateIndexes[i%2 + 1]

    end
end;

function test1()
    n = 500000
    TestDict1 = Dict{Int64, Dict{String, Any}}();
    TestDict2 = Dict{Int64, Dict{String, String}}();
    TestDict3 = Dict{Int64, Dict{String, Int64}}();

    testdf = DataFrame(state = String[], infectedNeighbours=Int64[])

    array_dict1 = convert.(Int64, zeros(n))
    array_dict2 = convert.(Int64, zeros(n))
    array_df = convert.(Int64, zeros(n))

    states = ["I", "R"]
    states1 = ['I', 'R']

    for i::Int64 in 1:n
        TestDict1[i] = Dict("state"=>"S", "infectedNeighbours"=>i)
        TestDict2[i] = Dict("state"=>"S")
        TestDict3[i] = Dict("infectedNeighbours"=>i)

        push!(testdf, ("S", i))
    end

    for x in 1:10
        for i::Int64 in 1:n
            @inbounds array_dict1[i] = TestDict1[i]["infectedNeighbours"]
            @inbounds array_dict2[i] = TestDict3[i]["infectedNeighbours"]
            @inbounds array_df[i] = testdf[i, :infectedNeighbours]

        end

        for i::Int64 in 1:n
            @inbounds TestDict1[i]["state"] = states[i%2 + 1]
            @inbounds TestDict2[i]["state"] = states[i%2 + 1]
            @inbounds testdf[i, :state] = states[i%2 + 1]
        end

        for i::Int64 in 1:n
            @inbounds TestDict1[i]["infectedNeighbours"] += 1
            @inbounds TestDict3[i]["infectedNeighbours"] += 1
            @inbounds testdf[i, :infectedNeighbours] += 1
        end
    end
end;

@profiler for j in 1:2
    test1()
end

@time test1()

@profiler for j in 1:10
    test2()
end

@time test2()
