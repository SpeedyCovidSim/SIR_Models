import sys
from numpy.core.function_base import linspace
import pandas as pd
sys.path.append("/Users/joeltrent/Documents/GitHub")
sys.path.append("/Users/joeltrent/networkcontagion/conditioning")
sys.path.append("/Users/joeltrent/networkcontagion/conditioning/lib")
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from ccandu.ccandu import conditional_composer as cc
from plot_conditioned import make_conditioned_plot as mcp
from plot_daily_conditioned import make_conditioned_plot as mcpDaily
# import get_GP_model as ggm


def get_cases(dir, net=True):
    # if net:
    #     ref = pd.read_csv(f"{dir}/config_{ind}.stats.csv", index_col=0)
    #     detected = np.isfinite(ref['Time to Detection']).index
    df = pd.read_csv(f"{dir}", index_col=[0,1])
    df.columns = df.columns.astype(int)
    if net:
        df = df.loc[detected, 0:60]
    else:
        df = df.loc[:, 0:60]
        detected = [0]
    conf_cases = df.xs("Known Cases", level='metric').fillna(method='ffill', axis='columns').values
    cumul_cases = df.xs("Cumulative Cases", level='metric').fillna(method='ffill', axis='columns').values

    return conf_cases, cumul_cases, detected, df

def condition(dir):

    net = False
    daily = False
    t_coeff = ggm.performTukeyapprox(dir, daily, net)
    print(f'transformation coefficient: {t_coeff}')
    output_dict = ggm.condition_linear_sample(dir, t_coeff, daily, net, full_return=True)

    # August 2021
    data = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566, 615, 690, 739, 767, 787, 807, 827, 848, 863, 876]
    # data = [1, 9, 10, 12, 19, 23, 33, 41, 62, 68, 70, 82, 83, 53, 49, 75, 49, 28, 20, 20, 20, 21, 15, 13]

    # data = [1,5,9,25,45,72,108,145,205,274,353,429,508,562,627,686,731,759]
    # data = [1,4,4,16,20,27,36,37,62,66,79,76,79,54,62,59,45,28,22,24,15,26,13,14]

    # August 2020
    # data = [1, 4, 17, 29, 34, 47, 59, 68, 78, 83, 85, 92, 94, 101]
    # data = [1, 3, 13, 12, 5, 14, 12, 9, 10, 5, 2, 7, 2, 7]
    # data = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193]

    # cond_runs = mcp(output_dict, data, net, band = 'quant', net_mod = 'filter', error=False, Reff=False, save=False)

    if daily:
        cond_runs = mcpDaily(output_dict, data, net, band = 'quant', net_mod = 'filter', error=False, Reff=False, save=False)
        return cond_runs
    else:
        cond_runs = mcp(output_dict, data, net, band = 'quant', net_mod = 'filter', error=False, Reff=False, save=False)
        return cond_runs

def getGPBands(conf_cases, data):

    # net = False
    # daily = False
    # t_coeff = ggm.performTukeyapprox(dir, daily, net)
    # print(f'transformation coefficient: {t_coeff}')
    # output_dict = ggm.condition_linear_sample(dir, t_coeff, daily, net, full_return=True)

    # input_data = pd.read_csv(f"{dir}/parameters.csv", index_col=0)
    # config = input_data.iloc[0]
    # i = config.name

    # # conf_cases, cumul_cases, detected, df = get_cases(i, dir, net)

    # conf_cases, cumul_cases, detected, df = ggm.get_cases(i, dir, net)

    # have to transform conf_cases


    # August 2021
    # data = np.array([[1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566, 615, 690, 739, 767, 787, 807, 827, 848, 863, 876, 887, 910, 930, 966]])
    # data = [1, 9, 10, 12, 19, 23, 33, 41, 62, 68, 70, 82, 83, 53, 49, 75, 49, 28, 20, 20, 20, 21, 15, 13]

    # data = [1,5,9,25,45,72,108,145,205,274,353,429,508,562,627,686,731,759]
    # data = [1,4,4,16,20,27,36,37,62,66,79,76,79,54,62,59,45,28,22,24,15,26,13,14]

    # August 2020
    # data = [1, 4, 17, 29, 34, 47, 59, 68, 78, 83, 85, 92, 94, 101]
    # data = [1, 3, 13, 12, 5, 14, 12, 9, 10, 5, 2, 7, 2, 7]
    # data = [1,4,17,29,34,47,59,68,78,83,85,92,94,101,108,111,116,122,131,135,139,141,145,149,151,154,159,161,165,171,172,174,176,177,178,179,179,184,184,185,187,187,188,191,191,192,192,192,192,192,192,192,192,193,193]

    # cond_runs = mcp(output_dict, data, net, band = 'quant', net_mod = 'filter', error=False, Reff=False, save=False)

    quartile_x = list(range(len(data[0]), len(conf_cases[0])))

    lower_95, upper_95 = cc.construct_bands(conf_cases, list(range(len(data[0]), len(conf_cases[0]))), data, list(range(0,len(data[0]))), prob=0.95, transformation_type='sk_quantile')

    lower_50, upper_50 = cc.construct_bands(conf_cases, list(range(len(data[0]), len(conf_cases[0]))), data, list(range(0,len(data[0]))), prob=0.5, transformation_type='sk_quantile')
    # lower_50, upper_50 = [], []

    return lower_95, upper_95, lower_50, upper_50, quartile_x

def conditionABC(conf_cases, data):

    filtered_ABC = cc.filter_ABC(realisations=conf_cases, obs_indices=list(range(0, len(data[0]))),
        data=data, q_tol=0.05, return_indices=False)

    indexes = cc.filter_ABC(realisations=conf_cases, obs_indices=list(range(0, len(data[0]))),
        data=data, q_tol=0.05, return_indices=True)

    t = list(range(0, len(data[0])))

    # Display=False
    # if Display:
    #     sns.set()
    #     sns.set_style('ticks')
    #     sns.set_color_codes('pastel')
    #     f = plt.figure(figsize=[6,5],dpi=300)

    #     for i in range(0, len(indexes[1])):
    #         plt.plot(list(range(0, len(filtered_ABC[i,:]))), filtered_ABC[i,:], "b-", lw=1, alpha = 1.0)

    #     plt.plot(t, data[0], "k-", label="data", lw=2, alpha = 1.0)

    #     plt.legend(loc = 'lower right')

    #     # required to display graph on plots.
    #     plt.show()

    indexesJulia = np.array(indexes[1]) + 1
    return indexesJulia

def conditioningJulia(dir, data):

    net=False
    # input_data = pd.read_csv(f"{dir}/parameters.csv", index_col=0)
    # config = input_data.iloc[0]
    # i = config.name

    # conf_cases, cumul_cases, detected, df = get_cases(i, dir, net)

    conf_cases, cumul_cases, detected, df = get_cases(dir, net)

    conf_cases = np.nan_to_num(conf_cases)

    # data = np.array([[1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566, 615, 690, 739, 767, 787, 807, 827, 848, 863, 876, 887, 910, 930, 966]])
    # print(data)
    data = np.array([data])
    # print(data)

    indexesJulia = conditionABC(conf_cases, data)

    lower_95, upper_95, lower_50, upper_50, quartile_x = getGPBands(conf_cases, data)

    return indexesJulia, lower_95, upper_95, lower_50, upper_50, quartile_x


# dir = '/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs'
# # data = [1, 10, 20, 32, 51, 74, 107, 148, 210, 278, 348, 430, 513, 566, 615, 690, 739, 767, 787, 807, 827, 848, 863, 876, 887, 910, 930, 966]
# indexesJulia, lower_95, upper_95, lower_50, upper_50 = conditioningJulia(dir)
# print(lower_95)


# dir = '/Users/joeltrent/Documents/GitHub/SIR_Models/August2020OutbreakFit/CSVOutputs'
# print()

# DF = pd.DataFrame(condition(dir))

# # save the dataframe as a csv file
# DF.to_csv('/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/indexes_ensemble_config_7_10Sep_cumulative.csv', header=False, index=False)

# DF.to_csv('/Users/joeltrent/Documents/GitHub/SIR_Models/August2020OutbreakFit/CSVOutputs/indexes_ensemble_6Sep_MatchMike.csv', header=False, index=False)


# dir = '/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs'

# DF = pd.DataFrame(conditionABC(dir))
# DF.to_csv('/Users/joeltrent/Documents/GitHub/SIR_Models/August2021Outbreak/CSVOutputs/indexes_ensemble_config_11_13Sep_ABC.csv', header=False, index=False)

# lower_95, upper_95 = getGPBands(dir)

# print(lower_95)
