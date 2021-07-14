using TrackingHeaps, DataStructures

struct event
    time::Union{Float64, Int64}
    isRecovery::Bool # false is infection event, true is recovery event
    parentID::Int64
    sSaturation::Union{Float64, Int64} # ∈ [0,1]
end

function Base.isless(x::event, y::event)::Bool
    return  x.time < y.time
end


t_max = 100

eventList = [event(1, true, 1, 1.0), event(2, false, 2, 1.0)]

eventList2 = [event(t_max+i,false,0,1.0) for i=1:10]



h = TrackingHeap(event, S=NoTrainingWheels, O=MinHeapOrder, N = 3, init_val_coll=eventList)
track!(h, event(100,true,0,1.0))

# if event at key 'x' is an infection event, update that key to be a recovery event
# for the individual infected
# if event at key 'x' is a recovery event,
update!(h, 1, event(20, true, 1,1.0))


h2 = TrackingHeap(event, S=NoTrainingWheels, O=MinHeapOrder, N = 3, init_val_coll=eventList2)

top(h2)
a = top(h)

h

h = MutableBinaryHeap{event, DataStructures.FasterForward}()

push!(h, event(1, true, 2, 0.5))
push!(h, event(2, false, 3, 0.5))

pop!(h)
