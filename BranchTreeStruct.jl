#=

Class tree
 - know the top node(s) (seed cases)

struct of node

    - Node's know their children (pointer to?)
    - Nodes have an id - refers to their attributes in dataframe
    - Nodes have their time of infection event
    - Nodes know their parent?

Thinning function
search = ?heap? of current children of the nodes we have just looked at (e.g. children of seed cases
    in iteration 1), sorted by infection time.

    while search not empty
        target = pop(search)

        if target thinned
            remove target from tree (remove it from parent's child list)
        else
            (not thinned) add children to search, sorted by their infection times
            if children not yet simulated for this node
                simulate their children and add to tree and search

=#
mutable struct Node
    caseID::Int64
    infection_time::Union{Float64,Int64}
    parent::Union{Base.RefValue{Node},Base.RefValue{Nothing}}
    children::Array{Node,1}
end

struct Tree
    rootNodes::Array{Node,1}
end


node1 = Node(1, 10, Ref(nothing), []);
node2 = Node(2, 11, Ref(node1), []);
node3 = Node(3, 12, Ref(node1), []);

node1.children = [node2, node3];
println(node1.children)

infectiontree = Tree([node1]);

node4 = Node(4,13, Ref(node1), []);
push!(node1.children, node4);

println(infectiontree)



x = (a=1, b=2)

x.a
x.b


d = [1,2,3]
deleteat!(d, 2)

deleteat!(node1.children, 1)
