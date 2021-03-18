
module networkFunctions
    using LightGraphs, MetaGraphs, StatsBase

    export initialiseNetwork!, calcHazard!, incrementInfectedNeighbors!, changeState!,
        incrementStateTotals!

    function initialiseNetwork!(network, infectionProp, states, stateEvents, eventHazards)
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
        # states[1] will always be S/the initial state
        for i in vertices(network)
            set_props!(network, i, Dict(:initInfectedNeighbors=>0, :state=>states[1], :stateIndex=>1))

        end

        # randomly choose individuals to be infected
        infectedVertices = sample(1:numVertices, Int(ceil(infectionProp *
            numVertices)), replace = false)

        # Set them as infected and increment numInfectedNeighbors for their localNeighbourhood
        # by 1
        for i in infectedVertices
            set_prop!(network, i, :state, "I")
            set_prop!(network, i, :stateIndex, findfirst(states .== "I"))

            incrementInfectedNeighbors!(network, i)

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

        #I_total = length(infectedVertices)
        #S_total = numVertices - I_total
        #R_total = 0

        stateTotals = zeros(length(states))
        stateIncrementor = zeros(length(states))

        # count the num of vertices in each state
        for i in vertices(network)
            stateTotals += states .== get_prop(network, i, :state)
        end

        set_props!(network, Dict(:stateTotals=>stateTotals, :states=>states,
            :stateEvents=>stateEvents, :eventHazards=>eventHazards,
            :population=>numVertices, :stateIncrementor=>stateIncrementor))

        return
    end

    function calcHazard!(network, alpha, beta, updateOnly = false, hazards = [],
        vertexIndex = 0, event = "infected", events = ["I", "R"])
        #=
        Inputs
        network : a undirected graph from LightGraphs and MetaGraphs
                  containing our population of interest
        alpha   : probability of infected person recovering [0,1]
        beta    : probability of susceptible person being infected [0,1]
        hazards : calculated hazard for each individual (vertex) in the network
        updateOnly : bool, true means only update the hazards array
        vertexIndex : the individual (vertex) the event occured to
        event       : string containing either "recovered" or "infected"
        events      : an array of strings = ["infected", "recovered"]

        Outputs
        hazards : calculated hazard for each individual (vertex) in the network
                : or nothing if updateOnly as it operates in place on hazards array
        =#

        if updateOnly
            # change hazard of affected individual
            # if infected, hazard is alpha, otherwise 0 as they've recovered
            hazards[vertexIndex] = alpha * (event == events[1])

            if event == events[2]
                increment = -1 * beta
            else
                increment = 1 * beta
            end

            # multithread hazard calculation to speed up
            # if num neighbors is low, this will actually slow it down
            Threads.@threads for i in neighbors(network, vertexIndex)
                # only update hazards for those that are susceptible
                if get_prop(network, i, :state) == "S"
                    hazards[i] = hazards[i] + increment
                end
            end

            #fil1 = filter_vertices(network[neighbors(network, vertexIndex)], :state, "S")

        else # initialisation

            # preallocate hazards array
            hazards = zeros(get_prop(network, :population))

            # calculate hazard at i
            # use multithreading to speed up
            Threads.@threads for i in vertices(network)
                if get_prop(network, i, :state) == "S"
                    hazards[i] = beta * get_prop(network, i, :initInfectedNeighbors)
                elseif get_prop(network, i, :state) == "I"
                    hazards[i] = copy(alpha)

                    #else # person is recovered and their hazard is zero
                end
            end
            return hazards
        end

        return nothing
    end

    function incrementInfectedNeighbors!(network, vertexIndex)
        #=
        Inputs
        network     : a undirected graph from LightGraphs and MetaGraphs
                      containing our population of interest
        vertexIndex : the individual (vertex) the event occured to
        event       : string containing either "recovered" or "infected"
        events      : an array of strings = ["infected", "recovered"]

        Outputs
        nothing     : works in place on the network
        =#

        for i in neighbors(network, vertexIndex)
            set_prop!(network, i, :initInfectedNeighbors, get_prop(network, i, :initInfectedNeighbors) + 1)
        end
        return nothing
    end

    function changeState!(network, vertexIndex, state, newStateIndex)
        #=
        Inputs
        network     : a undirected graph from LightGraphs and MetaGraphs
                      containing our population of interest
        vertexIndex : the individual (vertex) the state change occured to
        state       : state to change to

        Outputs
        nothing     : works in place on the network
        =#

        set_prop!(network, vertexIndex, :state, state)
        set_prop!(network, vertexIndex, :stateIndex, newStateIndex)

        #=
        if state == "I"
            set_prop!(network, vertexIndex, :state, "I")
        elseif state == "R"
            set_prop!(network, vertexIndex, :state, "R")
        end
        =#
        return nothing
    end

    function incrementStateTotals!(network, prevStateIndex, newStateIndex)
        #=
        Inputs
        network         : a undirected graph from LightGraphs and MetaGraphs
                          containing our population of interest
        prevStateIndex  : the previous state Index of the individual updated
        newStateIndex   : the new state Index of the individual updated
                        Indexing w/ respect to State indexing property of network

        Outputs
        nothing     : works in place on the network
        =#

        # stateIncrementor is an array of zeros
        incrementorArray = copy(get_prop(network, :stateIncrementor))
        incrementorArray[prevStateIndex] = -1
        incrementorArray[newStateIndex] = 1

        set_props!(network, Dict(:stateTotals=>get_prop(network, :stateTotals) + incrementorArray))

        return nothing
    end

end
