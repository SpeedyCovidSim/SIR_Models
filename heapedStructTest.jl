using DataStructures
using TrackingHeaps

struct event
    time::Union{Float64, Int64}
    recoveryEvent::Bool
    parentID::Int64
end

mutable_binary_minheap(event)

MutableBinaryMinHeap(event)

MutableBinaryMinHeap()

h = TrackingHeap(Float64, S=NoTrainingWheels, O=MinHeapOrder, N = 3, init_val_coll=tau_i)
