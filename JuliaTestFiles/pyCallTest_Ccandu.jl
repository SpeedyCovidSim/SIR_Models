using PyCall

py"""
import sys
sys.path.append("/Users/joeltrent/Documents/GitHub/SIR_Models")
from conditionEnsemble import conditioningJulia
"""
conditionEnsemble = py"conditioningJulia"


dir = "/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs"

filterVector, lower_95, upper_95, lower_50, upper_50 = conditionEnsemble(dir)

print(lower_95)
