#=
This is a test of the SparseArray and SparseArrayKit packages vs their dense
counterparts.

 -  Speed of access and summing across dimensions is our most important questions
    also memory use
 -  once initialised we won't alter the values in the matrix.
 -  3D sparse matrix is desired for summing across contexts as well. Test versus a
    dictionary of sparse Arrays.

Author: Joel Trent
=#

using SparseArrays, BenchmarkTools
# using SparseArrayKit

push!(LOAD_PATH, "/Users/joeltrent/Documents/GitHub/")
using ArrayIteration

n = 100000;
density = 1/n; # Float between 0 and 1

# sparse Dict init
sparseDict = Dict{Int, SparseMatrixCSC{Float64,Int64}}()

# generate matrix
testMatrix_sparse = sprand(n,n,density); # sparse version
# testMatrix_dense = Array(testMatrix_sparse); # dense version

# test answers are the same
round(sum(testMatrix_sparse), sigdigits=10) == round(sum(testMatrix_dense), sigdigits=10)
round(sum(testMatrix_sparse[:,1]), sigdigits=10) == round(sum(testMatrix_dense[:,1]), sigdigits=10)

sparseDict[1] = testMatrix_sparse;
sparseDict[2] = sprand(n,n,density);

sum(sparseDict[1], dims = 1) + sum(sparseDict[2], dims = 1)

sparseDict[1] + sparseDict[2]

isState = convert.(Bool,zeros(n, 5))
isState[:,2] .= true

# element-wise multiplication works. Addition works. JULIA IS COL-WISE - row-WISE
# is much slower
@benchmark sum(sparseDict[1][:,2] .* isState[:,2]) # 10^3 times slower than .* 1.0
@benchmark sum(sparseDict[1][:,2] .* 1.0)
sparseDict[1][:,2] .+ 1.0

sum(sparseDict[1][:,2] .* isState[:,2])

println("lol")

nzrange(sparseDict[1][:,2], 1)
rowvals(sparseDict[1])

# this is not very quick
# could use some of the problem structure to make this faster - we know the connections
# within a given context using neighbours(context) in the graph. And these have
# to be entries in the sparseDict
function testSparseMult_fast(sparseDict, isState, context::Int, individual::Int, stateIndex::Int)
    sum = 0.0
    for I in each(index(stored(sparseDict[context][:,individual])))
        sum += sparseDict[context][I,individual] * isState[I, stateIndex]
    end
end

@benchmark testSparseMult_fast(sparseDict, isState, 1, 2, 2)

sparseDict[1][:,2]
neighborArray = [8807, 51039, 72561]
# stored(sparseDict[1][:,2])

# create a more true to our case example - this is very good. Very fast.
# much faster than element-wise multiplication of entire isState array.
function testSparseMult_Graph(sparseDict, isState, context::Int, individual::Int,
    stateIndex::Int, neighborArray::Array{Int,1})
    sum = 0.0
    for i in neighborArray
        if i != individual
            sum += sparseDict[context][i,individual] * isState[i, stateIndex]
        end
    end
end

@benchmark testSparseMult_Graph(sparseDict, isState, 1, 2, 2, neighborArray)

################## This is from the package test cases ##################
#=
generate a whole bunch of random contractions, compare with the dense result
=#
function randn_sparse(T::Type{<:Number}, sz::Dims, p = 0.5)
    a = SparseArray{T}(undef, sz)
    for I in keys(a)
        if rand() < p
            a[I] = randn(T)
        end
    end
    return a
end

MAX_DIM = 4;
MAX_LEGS = 3;

dims = ntuple(l->rand(1:MAX_DIM), MAX_LEGS)
ar = randn_sparse(Float64, dims)
ac = randn_sparse(ComplexF64, dims)

ar[1,:,1]

nonzero_values(ar)

n = 1000000;
size = (n,n,3)

a = SparseArray{Float64}(undef, size)

density = 1/n;
twoD_sparse = sprandn(n,n, density)

################## This is from the package test cases ##################

#=
As far as I can tell, there is no way of initialising the 3d version of a sparse
Array with pre-existing values. It's only ever using 'undef' which means it's quite
slow to fill it. I.e. the below expression does not work. Resultantly, I will
use a dictionary implementation which contains a sparse array for each context
=#
# SparseArray{Float64}([1.0,2.0,3.0], (1,1,3))
