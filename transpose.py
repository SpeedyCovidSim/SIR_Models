import pandas as pd
def transposeCSV(origPath, newPath):
    # origPath = 'BP_csv/BP2021ensemble_cumulativecases.csv'
    # newPath = 'BP_csv/config_1.csv'
    pd.read_csv(origPath, header=None).T.to_csv(newPath, header=False, index=False)
