
module networkFunctions
    using LightGraphs, MetaGraphs, StatsBase

    export initialiseNetwork, calcHazard, incrementInfectedNeighbors, changeState

    function initialiseNetwork(network, infectionProp)
        #=
        Inputs
        network       : a undirected graph from LightGraphs and MetaGraphs
                        containing our population of interest
        infectionProp : proportion of people to be infected [0, 1)

        Outputs
        network       : works in place on the network. network is initialised
        S_total       : Num people susceptible to infection
        I_total       : Num people infected
        R_total       : Num people recovered
        =#

        # return num vertices
        numVertices = nv(network)

        # Initialise all states as "S", all numInfectedNeighbors as zero
        for i in 1:numVertices
            set_prop!(network, i, :numInfectedNeighbors, 0)
            set_prop!(network, i, :state, "S")
        end

        # randomly choose individuals to be infected
        infectedVertices = sample(1:numVertices, Int(ceil(infectionProp *
            numVertices)), replace = false)

        # Set them as infected and increment numInfectedNeighbors for their localNeighbourhood
        # by 1
        for i in infectedVertices
            set_prop!(network, i, :state, "I")

            incrementInfectedNeighbors(network, i, "infected")

            #=
            localNeighbourhood = neighbors(network, i)
            for j in localNeighbourhood
                set_prop!(network, j, :numInfectedNeighbors, get_prop(network, j, :numInfectedNeighbors) + 1)

                #= Technically only matters if they are susceptible, but will write
                like this^ to make more future proof (additional states)
                if  get_prop(network, j, :state) == "S"

                end=#
            end
            =#
        end

        I_total = length(infectedVertices)
        S_total = numVertices - I_total
        R_total = 0

        return S_total, I_total, R_total
    end

    function calcHazard(network, alpha, beta)
        #=
        Inputs
        network : a undirected graph from LightGraphs and MetaGraphs
                  containing our population of interest
        alpha   : probability of infected person recovering [0,1]
        beta    : probability of susceptible person being infected [0,1]

        Outputs
        hazards : calculated hazard for each individual (vertex) in the network
        =#

        # return num vertices
        numVertices = nv(network)

        hazards = zeros(numVertices)

        for i in 1:numVertices
            if get_prop(network, i, :state) == "S"
                hazards[i] = beta * get_prop(network, i, :numInfectedNeighbors)
            elseif get_prop(network, i, :state) == "I"
                hazards[i] = copy(alpha)

                #else # person is recovered and their hazard is zero
            end
        end
        return hazards
    end

    function incrementInfectedNeighbors(network, vertexIndex, event)
        #=
        Inputs
        network     : a undirected graph from LightGraphs and MetaGraphs
                      containing our population of interest
        vertexIndex : the individual (vertex) the event occured to
        event       : string containing either "recovered" or "infected"

        Outputs
        nothing     : works in place on the network
        =#

        if event == "recovered"
            increment = -1
        else
            increment = 1
        end

        for i in neighbors(network, vertexIndex)
            set_prop!(network, i, :numInfectedNeighbors, get_prop(network, i, :numInfectedNeighbors) + increment)
        end
        return nothing
    end

    function changeState(network, vertexIndex, state)
        #=
        Inputs
        network     : a undirected graph from LightGraphs and MetaGraphs
                      containing our population of interest
        vertexIndex : the individual (vertex) the state change occured to
        state       : state to change to

        Outputs
        nothing     : works in place on the network
        =#

        if state == "I"
            set_prop!(network, vertexIndex, :state, "I")
        elseif state == "R"
            set_prop!(network, vertexIndex, :state, "R")
        end

        return nothing
    end
end
