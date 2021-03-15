#=
This is a test of the network functionality

This is a code base for a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
=#

using LightGraphs, GraphPlot, MetaGraphs, BenchmarkTools

# import required modules
push!( LOAD_PATH, "./" )    #Set path to current
using networkFunctions


##
# Return a Watts-Strogatz small world random graph with n vertices, each with
# expected degree k
beta = 1
n = 10000
k = 100
#network = MetaGraph(watts_strogatz(n, k, beta))
network = MetaGraph(complete_graph(n))

gplot(network, nodelabel=1:n)



# tell me which neighbors you have
network[neighbors(network,1)]


network[2, :index]


# get a property of a vertex
#get_prop(network, 2, :state)

initialiseNetwork(network, 0.05)

filter_vertices(network[neighbors(network, 1)], :state, "S")


for i in filter_vertices(network[neighbors(network, 2)], :state, "S")
    set_prop!(network, i, :state, "R")
end

get_prop(network, 5, :state)


function hazards1(hazards, alpha, beta)
    # calculate hazard at i
    for i in filter_vertices(network, :state, "S")
        hazards[i] = beta * get_prop(network, i, :initInfectedNeighbors)

    end
    hazards = hazards + (hazards .== 0) * alpha
    return hazards
end

function hazards2(hazards, alpha, beta)
    for i in vertices(network)
        if get_prop(network, i, :state) == "S"
            hazards[i] = beta * get_prop(network, i, :initInfectedNeighbors)
        elseif get_prop(network, i, :state) == "I"
            hazards[i] = copy(alpha)

            #else # person is recovered and their hazard is zero
        end
    end
    return hazards
end


alpha, beta = 1,2

h1Times = []
h2Times = []

for j in 1:1000
    hazards = zeros(get_prop(network, :population))
    push!(h1Times, @elapsed hazards1(hazards, alpha, beta))
    hazards = zeros(get_prop(network, :population))
    push!(h2Times, @elapsed hazards2(hazards, alpha, beta))
end

tMean = zeros(Float64,length(100),2)
tMedian = zeros(Float64,length(100),2)

tMean[1,:] = [mean(h1Times), mean(h2Times)]
tMedian[1,:] = [median(h1Times), median(h2Times)]


# preallocate hazards array
hazards = zeros(get_prop(network, :population))

@elapsed hazards = hazards1(hazards, alpha, beta)

# preallocate hazards array
hazards = zeros(get_prop(network, :population))

@time hazards = hazards2(hazards, alpha, beta)

alpha = 0.5
beta = 1
calcHazard(network, alpha, beta)



#=
# use vertex values as indices
for i in 1:n
    set_prop!(smallWorldG, i, :name, i)
end
set_indexing_prop!(smallWorldG, :name)
Set(Symbol[:name])

smallWorldG[1, :name]
=#


if false
    # size of graph
    n = 5
    e = 10
    G = SimpleGraph(n,e) # graph with n vertices, and e edges

    # create a complete undirected graph
    completeG = complete_graph(n)


    # Create a random undirected regular graph with n vertices, each with degree k.
    k = 4
    randomG = random_regular_graph(n, k)

    # Return a Watts-Strogatz small world random graph with n vertices, each with
    # expected degree k
    beta = 1
    smallWorldG = watts_strogatz(n, k, beta)

    # Creates a Turán Graph, a complete multipartite graph with n vertices and r partitions.
    r = 2
    turanG = turan_graph(n, r)

    #=
    # make a triangle
    add_edge!(G, 1, 2)
    add_edge!(G, 1, 3)
    add_edge!(G, 2, 3)
    =#
    gplot(G, nodelabel=1:n)
    gplot(completeG, nodelabel=1:n)
    gplot(randomG, nodelabel=1:n)
    gplot(smallWorldG, nodelabel=1:n)
    gplot(turanG, nodelabel=1:n)

    collect(edges(G))



end
