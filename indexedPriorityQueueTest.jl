
using TrackingHeaps

tau_i = Float64[8 4 5 6 10 3.5 4 5]

#=
Builds heap of type Float64
using init_val_coll alllows inputting heap values directly. Will sort automatically
NoTrainingWheels means that we promise to always use allowed indexes
MinHeapOrder means smallest time is at top

N refers the arity / degree of the tree. Default is a binary tree (N=2), but
should get speedups for N=3 (and maybe even higher) if we have to sort Inf values
a lot. Worth trying different values.
=#
h = TrackingHeap(Float64, S=NoTrainingWheels, O=MinHeapOrder, N = 3, init_val_coll=tau_i)

println(top(h))

# returns a pair. The tracker of the value and the value
a = top(h)

a[1]

a[2]

# automatically updates the ordering
tracker = track!(h, 1.0)
a = top(h)[1]

tracker2 = track!(h, Inf)
a = top(h)


# 99% sure this implements the update algorithm from Gibson and Bruck 2000, which
# we need for the NextReactionMethod
update!(h, tracker, 2.0)

# returns the value of the item in the heap.
getindex(h, tracker)
getindex(h, 10)


length(h)
