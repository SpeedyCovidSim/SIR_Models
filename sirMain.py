'''
This is a code base for the main function of a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
'''

from sirModels import gillespieDirect2Processes
from plots import plot
import numpy as np
import random

def main():
    # testing the gillespieDirect2Processes function

    # Get same thing each time
    random.seed(1)

    # initialise variables
    N = np.array([5, 10, 50, 100,1000,10000])

    S_total = N - 1
    I_total = np.ones(len(N))
    R_total = np.zeros(len(N))

    t_max = 200
    alpha = 0.4
    beta = 0.001

    # iterate through populations
    for i in range(len(N)):
        t, S, I, R = gillespieDirect2Processes(t_max, S_total[i], I_total[i],
            R_total[i], alpha, beta, N)

        # plot and export the simulation
        outputFileName = f"pythonGraphs/SIR_Model_Pop_{N[i]}"
        plot(t, [S, I, R], N[i], alpha, beta, outputFileName, Display=False)

if __name__=="__main__":
    main()