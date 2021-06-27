module bipartiteInitFunctions

    using LightGraphs, DataFrames, StatsBase, SparseArrays

    export bipartiteGenerator!

    function bipartiteGenerator!(populationSize::Int, populationAgeRanges::Array{}, percentInAgeRanges::Array{Float64,1},
        numberOfContextType::Array{Int64,1}, contextTypes::Array{Int64,1}, networkVertex_df::AbstractDataFrame, beta)
        #=
        A function that creates a bipartite network given an input.
        This means:
            - Context vertices cannot be connected to one another
            - Individual vertices cannot be connected to one another directly, only
            through contexts.
            - Rule of every individual has a home (context 1)
            - Can only belong to one of each type of context. Mainly for homes and schools
            but could be for workplaces too.
            - will add rule for not belonging to school and work at same time depending
            on an age.

        Contexts are considered to be the last sum(numberContexts) vertexes, while
        the population is in the first 1:population vertexes of the network.
        ContextTypes are 1:number of contexts. Indiv has type 0

        Also, operates in place on networkVertex_df

        =#

        # this must be equal to the column length of networkVertex_df - although
        # we might consider the dataframe to be empty before hitting this function
        numVertices = sum(numberOfContextType) + populationSize

        bipartiteNetwork = SimpleGraph(numVertices)

        # init vertex type dataframe
        vertexTypeInit!(numVertices, populationSize, numberOfContextType, contextTypes,
            networkVertex_df)

        # add any additional attributes to individuals (e.g. their age which could
        # affect whether or not they are put into a school or workplace context)

        contextBetaInit!(numVertices, populationSize, numberOfContextType, contextTypes,
            networkVertex_df, beta)


        # create the bipartite representation - very simplified for the moment

        # add people to a given context. Rule of every individual has a home
        for index in 1:length(numberOfContextType)
            numOfContext = numberOfContextType[index]

            individuals = collect(1:populationSize)

            equalSplit = ceil(Int, populationSize/numOfContext)
            numToSample = equalSplit
            # numToSample::Int = sample(collect(equalSplit - )


            # DON'T DO IT THIS WAY. Setdiff is extremely expensive.
            # Instead loop through each individual and add to a random context.
            # maybe have a rejection step if we have a maximum allowed in a context.

            indivNotSampled = Array{Int8}(undef, populationSize)
            indivNotSampled .= 1

            for i in 1:numOfContext

                if numToSample <= length(individuals)
                    # most of the time is spent sampling. Worth trying to decrease the
                    # cost of this step.
                    sampledIndivs = sample(individuals, StatsBase.weights(indivNotSampled), numToSample, replace=false)

                else
                    sampledIndivs = copy(individuals)
                end

                addToContext!(bipartiteNetwork, sampledIndivs, index, i, numberOfContextType, populationSize)

                indivNotSampled[sampledIndivs] .= 0


                if i == numOfContext && contextTypes[index] == 1 # a home context

                    # add any indivs left to any context of current type - atm will
                    # just make the last context
                    addToContext!(bipartiteNetwork, individuals, index, i,
                        numberOfContextType, populationSize)

                end

            end
        end


        return bipartiteNetwork
    end;

    function addToContext!(bipartiteNetwork, sampledIndivs::Array{Int64,1}, contextTypeIndex,
        contextNumberInType, numberOfContextType, populationSize)

        contextVertex = populationSize + contextNumberInType + sum(numberOfContextType[1:(contextTypeIndex-1)])

        for individual::Int64 in sampledIndivs
            add_edge!(bipartiteNetwork, contextVertex, individual)
        end
        return nothing
    end;

    function contextBetaInit!(numVertices::Int, populationSize::Int,
        numberOfContextType::Array{Int64,1}, contextTypes::Array{Int64,1},
        networkVertex_df::AbstractDataFrame, beta)
        #=
        Operates in place on the networkVertex_df
        Adds the Beta column as a sparsevec (sparseArray/Vector) for different
        contexts. To begin with let us consider all contexts to have the same beta
        values.
        =#

        # init the column
        contextBeta = Array{Int64}(undef, numVertices-populationSize)

        contextBeta .= 1.0 * beta # beta::Float64

        indexes = cumsum(convert.(Int64, ones(numVertices-populationSize))) .+ populationSize

        sparseBeta = sparsevec(indexes, contextBeta)

        # sparsevec only stores values where needed - should save storage.
        networkVertex_df.beta = sparseBeta
        return nothing
    end;

    function vertexTypeInit!(numVertices::Int, populationSize::Int,
        numberOfContextType::Array{Int64,1}, contextTypes::Array{Int64,1},
        networkVertex_df::AbstractDataFrame)
        #=
        Operates in place on the networkVertex_df
        Adds the type column representing whether the vertex is a individual or one
        of a number of contexts. 0 if an individual, 1:number of context types, if
        a context.
        =#

        # init the column
        vertexType = Array{Int64}(undef, numVertices)

        vertexType[1:populationSize] .= 0

        for index::Int64 in 1:length(contextTypes)
            indexStart = populationSize + 1 + sum(numberOfContextType[1:(index-1)])
            indexEnd = indexStart + numberOfContextType[index] - 1
            vertexType[indexStart:indexEnd] .= contextTypes[index]
        end

        # add column to dataframe
        networkVertex_df.type = vertexType

        return nothing
    end;

end  # module bipartiteInitFunctions
