using BenchmarkTools

mutable struct NodeDict
    caseID::Int64
    infection_time::Union{Float64,Int64}
    parent::Union{Base.RefValue{NodeDict},Base.RefValue{Nothing}}
    children::Dict{Int64, NodeDict}
end

struct Tree
    rootNodes::Array{NodeDict,1}
end

mutable struct NodeArray
    caseID::Int64
    infection_time::Union{Float64,Int64}
    parent::Union{Base.RefValue{NodeArray},Base.RefValue{Nothing}}
    children::Array{NodeArray,1}
end

struct Tree2
    rootNodes::Array{Node,1}
end

#
# delete(nt::NamedTuple{names}, keys) where names =
#     NamedTuple{filter(x -> x ∉ keys, names)}(nt)
#
# c = 1
# c = (:1=node1, :2=node2)
# delete(c, a)

function dictTest()
    for i in 1:5000
        node1 = NodeDict(1, 10, Ref(nothing), Dict{Int64,NodeDict}());
        node2 = NodeDict(2, 11, Ref(node1), Dict{Int64,NodeDict}());
        node3 = NodeDict(3, 12, Ref(node1), Dict{Int64,NodeDict}());
        node4 = NodeDict(4, 13, Ref(node1), Dict{Int64,NodeDict}());
        node5 = NodeDict(5, 11, Ref(node1), Dict{Int64,NodeDict}());
        node6 = NodeDict(6, 12, Ref(node1), Dict{Int64,NodeDict}());
        node7 = NodeDict(7, 13, Ref(node1), Dict{Int64,NodeDict}());

        node1.children = Dict{Int64,NodeDict}(2=>node2, 3=>node3, 4=>node4, 5=>node5, 6=>node6, 7=>node7)

        delete!(node1.children, 7)

        childrenArray = collect(values(node1.children))

    end
end

function arrayTest()
    for i in 1:5000
        node1 = NodeArray(1, 10, Ref(nothing), []);
        node2 = NodeArray(2, 11, Ref(node1), []);
        node3 = NodeArray(3, 12, Ref(node1), []);
        node4 = NodeArray(4, 13, Ref(node1), []);
        node5 = NodeArray(2, 11, Ref(node1), []);
        node6 = NodeArray(3, 12, Ref(node1), []);
        node7 = NodeArray(4, 13, Ref(node1), []);

        node1.children = [node2, node3, node4, node5, node6, node7]

        for i in 1:length(node1.children)
            if node1.children[i].caseID == 6
                deleteat!(node1.children, 6)
                break
            end
        end

        childrenArray = node1.children

    end
end

dictTest()
arrayTest()

@benchmark dictTest()
@benchmark arrayTest()

@profiler dictTest()
@profiler arrayTest()
