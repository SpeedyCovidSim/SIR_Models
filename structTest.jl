## Switch to mutable struct from a nested dictionary

mutable struct Model
    iteration::Int64
    current_time::Float64
    modelName::String
end

model = Model(1, 10.0, "testStruct")

model.iteration = 2


exp(-1)
