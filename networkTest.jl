#=
This is a test of the network functionality

This is a code base for a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
=#

using LightGraphs, GraphPlot, MetaGraphs

# import required modules
push!( LOAD_PATH, "./" )    #Set path to current
using networkFunctions


##
# Return a Watts-Strogatz small world random graph with n vertices, each with
# expected degree k
beta = 1
n = 5
k = 2
network = MetaGraph(watts_strogatz(n, k, beta))

gplot(network, nodelabel=1:n)



# tell me which neighbors you have
neighbors(network,1)

# get a property of a vertex
#get_prop(network, 2, :state)

S_total, I_total, R_total = initialiseNetwork(network, 0.6)

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
