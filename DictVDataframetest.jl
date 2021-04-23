
using DataFrames

function test()
    n = 1000000
    TestDict1 = Dict{Int64, Dict{String, Any}}();
    TestDict2 = Dict{Int64, Dict{String, String}}();
    TestDict3 = Dict{Int64, Dict{String, Int64}}();

    testdf = DataFrame(state = String[], infectedNeighbours=Int64[])

    array_dict1 = convert.(Int64, zeros(n))
    array_dict2 = convert.(Int64, zeros(n))
    array_df = convert.(Int64, zeros(n))

    for i::Int64 in 1:n
        TestDict1[i] = Dict("state"=>"S", "infectedNeighbours"=>i)
        TestDict2[i] = Dict("state"=>"S")
        TestDict3[i] = Dict("infectedNeighbours"=>i)

        push!(testdf, ("S", i))
    end

    for i::Int64 in 1:n
        @inbounds array_dict1[i] = TestDict1[i]["infectedNeighbours"]
        @inbounds array_dict2[i] = TestDict3[i]["infectedNeighbours"]
        @inbounds array_df[i] = testdf[i, :infectedNeighbours]

    end

end

@profiler test()

@time test()
