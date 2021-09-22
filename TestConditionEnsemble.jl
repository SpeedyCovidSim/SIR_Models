push!( LOAD_PATH, "./" )
using ConditionEnsemble

dir = "/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/config_11_created11Sep.timeseries.csv"
data = [1,4,5,6,10]

filterVector, lower_95, upper_95, lower_50, upper_50 = conditionEnsemble(dir, data)

# println(lower_95)
