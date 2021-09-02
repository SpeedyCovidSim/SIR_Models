import sys
import pandas as pd
sys.path.append("/Users/joeltrent/Documents/GitHub")
sys.path.append("/Users/joeltrent/networkcontagion/conditioning")
sys.path.append("/Users/joeltrent/networkcontagion/conditioning/lib")
from ccandu.ccandu import conditional_composer as cc
from plot_conditioned import make_conditioned_plot as mcp
import get_GP_model as ggm


def condition(dir):

    net = False

    t_coeff = ggm.performTukeyapprox(dir, net)
    print(f'transformation coefficient: {t_coeff}')
    output_dict = ggm.condition_linear_sample(dir, t_coeff, net, full_return=True)

    # August 2021
    data = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566, 615, 690, 739]

    # data = [1,5,9,25,45,72,108,145,205,274,353,429,508,562,627,687]

    # August 2020
    # data = [1, 4, 17, 29, 34, 47, 59, 68, 78, 83, 85, 92, 94, 101]
    # data = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193]

    cond_runs = mcp(output_dict, data, net, band = 'quant', net_mod = 'filter', error=False, Reff=False, save=False)
    return cond_runs

dir = '/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs'
# print()

DF = pd.DataFrame(condition(dir))

# save the dataframe as a csv file
DF.to_csv('/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/indexes_ensemble_created2Sep.csv', header=False, index=False)
