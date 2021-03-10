'''
This is a code base for the main function of a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
'''

from sirModels import gillespieDirect2Processes
from plots import plotSIR
import numpy as np
import random

def main():
    # testing the gillespieDirect2Processes function

    # Get same thing each time
    random.seed(1)

    # initialise variables
    N = np.array([5, 10, 50, 100,1000,10000])

    S_total = N - np.ceil(0.05 * N)
    I_total = np.ceil(0.05 * N)
    R_total = np.zeros(len(N))

    t_max = 200
    alpha = 0.4
    beta = 10 / N

    # iterate through populations
    for i in range(len(N)):
        t, S, I, R = gillespieDirect2Processes(t_max, S_total[i], I_total[i],
            R_total[i], alpha, beta[i], N[i])

        # plot and export the simulation
        outputFileName = f"pythonGraphs/wellMixedDirectRandom/SIR_Model_Pop_{N[i]}"
        plotSIR(t, [S, I, R], alpha, beta[i], N[i], outputFileName, Display=False)

if __name__=="__main__":
    main()
