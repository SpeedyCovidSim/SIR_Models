'''
This is a code base for the main function of a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
'''

from sirModels import gillespieDirect2Processes
from matplotlib import pyplot as plt
import random
import numpy as np

def main():
    # testing the gillespieDirect2Processes function

    # Get same thing each time
    random.seed(1)

    # initialise variables
    N = 100

    S_total = N - 1
    I_total = np.ceil(0.05 * N)
    R_total = 0

    t_max = 200
    alpha = 0.4
    beta = 4 / N

    fig = plt.figure()
    
    # iterate through populations
    for i in range(100):
        t, S, I, R = gillespieDirect2Processes(t_max, S_total, I_total,
            R_total, alpha, beta, N)
        plt.plot(t, S, color="#82c7a5",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I, color="#f15e22",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, R, color="#7890cd",lw = 2, alpha=0.5,figure=fig)

    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Number of Individuals in State", fontsize=16)
    plt.title(f"Stochastic SIR models with a population size of {N}", fontsize=20)
    plt.show()
    

    N = 10000

    S_total = N - 1
    I_total = np.ceil(0.05 * N)
    R_total = 0

    t_max = 200
    alpha = 0.4
    beta = 4 / N

    fig = plt.figure()
    t, S, I, R = gillespieDirect2Processes(t_max, S_total, I_total,
            R_total, alpha, beta, N)
    S = np.array(S)/100
    I = np.array(I)/100
    R = np.array(R)/100
    plt.plot(t, S, label="Susceptible",color="#82c7a5",lw = 2, alpha=1,figure=fig)
    plt.plot(t, I, label="Infected",color="#f15e22",lw = 2, alpha=1,figure=fig)
    plt.plot(t, R, label="Recovered",color="#7890cd",lw = 2, alpha=1,figure=fig)
    plt.legend(fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Number of Individuals in State", fontsize=16)
    plt.title(f"ODE SIR model with a population size of 100", fontsize=20)
    plt.show()


if __name__=="__main__":
    main()