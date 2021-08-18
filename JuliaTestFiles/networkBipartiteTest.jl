#=
This is a test of the bipartite network functionality

Author: Joel Trent
=#


using LightGraphs, GraphPlot, MetaGraphs, BenchmarkTools, DataFrames, StatsBase, SparseArrays

push!( LOAD_PATH, pwd() )    #Set path to current
# joinpath(splitpath(pwd())[1:end-1]...)
using networkInitFunctions: initialiseNetwork!
using plotsPyPlot: plotSIRPyPlot
using bipartiteInitFunctions: bipartiteGenerator!


# # size of graph
# n = 5;
#
# # create a complete undirected graph
# completeG = complete_graph(n);
#
# # create a complete bipartite graph. all n2 nodes are connected to n1 nodes
# # let n1 be the number of contexts. Let n2 be the number of individuals
# # or other way round as desired
# n1 = 2;
# n2 = 4;
# completeBipart = complete_bipartite_graph(n1, n2);
# neighbors(completeBipart, 1)
#
#
# # create a multipartite graph - we don't want this.
# partitions = [1,2,1];
# multipartiteG = complete_multipartite_graph(partitions);
# gplot(multipartiteG, nodelabel=1:sum(partitions))
#
# # Creates a Turán Graph, a complete multipartite graph with n vertices and r partitions.
# r = 2;
# turanG = turan_graph(n, r);
#
#
# gplot(completeG, nodelabel=1:n)
# gplot(completeBipart, nodelabel=1:(n1+n2))
#
# gplot(turanG, nodelabel=1:n)
#
# collect(edges(G))
#
# bipartiteNetwork = SimpleGraph(10)
#
# add_edge!(bipartiteNetwork, 3, 1)
#
# collect(edges(bipartiteNetwork))
#
# gplot(bipartiteNetwork, nodelabel=1:length(vertices(bipartiteNetwork)))
#



N = 2000
t_max = 200
alpha = 0.4
beta = 10
gamma = 0
simType = "SIR_direct"
infectionProp = 0.05
bipartite_bool = true

networkVertex_df = DataFrame();
bipartiteNetwork = bipartiteGenerator!(N, Float64[], Float64[], [100,100,100], [1,2,3], networkVertex_df, beta)


networkVertex_df, network_dict, stateTotals, isState, model! = initialiseNetwork!(bipartiteNetwork, N, infectionProp, simType, alpha, beta, gamma, bipartite_bool, networkVertex_df)

time = @elapsed t, state_Totals = model!(t_max, bipartiteNetwork, alpha, beta, N, networkVertex_df, network_dict, stateTotals, isState)

S = state_Totals[1,:]
I = state_Totals[2,:]
R = state_Totals[3,:]
# D = state_Totals[4,:]
D = []

lengthStateTotals = length(state_Totals)
println("Length of state_Totals array is $lengthStateTotals")

println("Simulation has completed in $time")

Display = true
save = false
outputFileName = "BipartiteTest"

plotSIRPyPlot(t, [S,I,R,D], alpha, beta, N, outputFileName, Display, save)

# DataFrame happy to store a sparseArrays structure
# beta = sparsevec([6,7,8,9,10,11,12], [0.5,0.2,0.1,0.2,0.1,0.2,0.3])

# networkVertex_df.beta = beta

# gplot(bipartiteNetwork, nodelabel=1:length(vertices(bipartiteNetwork)))
