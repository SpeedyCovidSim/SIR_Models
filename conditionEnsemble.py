import sys
import pandas as pd
sys.path.append("/Users/joeltrent/Documents/GitHub")
sys.path.append("/Users/joeltrent/networkcontagion/conditioning")
sys.path.append("/Users/joeltrent/networkcontagion/conditioning/lib")
from ccandu.ccandu import conditional_composer as cc
from plot_conditioned import make_conditioned_plot as mcp
import get_GP_model as ggm


def transposeCSV(origPath, newPath):
    # origPath = 'BP_csv/BP2021ensemble_cumulativecases.csv'
    # newPath = 'BP_csv/config_1.csv'
    pd.read_csv(origPath, header=None).T.to_csv(newPath, header=False, index=False)

def condition(dir):

    net = False

    t_coeff = ggm.performTukeyapprox(dir, net)
    print(f'transformation coefficient: {t_coeff}')
    output_dict = ggm.condition_linear_sample(dir, t_coeff, net, full_return=True)

    # August 2021
    data = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566, 631]

    # August 2020
    #data = [1, 4, 17, 29, 34, 47, 59, 68, 78, 83, 85, 92, 94, 101]

    cond_runs = mcp(output_dict, data, net, band = 'quant', net_mod = 'filter', error=False, Reff=False, save=False)
    return cond_runs

dir = '/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs'
# print()

DF = pd.DataFrame(condition(dir))
  
# save the dataframe as a csv file
DF.to_csv('/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/indexes_31Aug.csv', header=False, index=False)