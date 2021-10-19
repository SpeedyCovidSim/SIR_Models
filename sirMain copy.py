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

    fig = plt.figure(figsize=[6,4],dpi=300)
    fig.tight_layout()
    
    # iterate through populations
    for i in range(50):
        t, S, I, R = gillespieDirect2Processes(t_max, S_total, I_total,
            R_total, alpha, beta, N)
        plt.plot(t, S, color="#777777",lw = 1,alpha=0.4, figure=fig)
        plt.plot(t, I, color="#c0d7f4",lw = 1,alpha=0.4, figure=fig)
        plt.plot(t, R, color="#f8b9b3",lw = 1,alpha=0.4, figure=fig)

    plt.xlabel("Time")
    plt.ylabel("Number of Individuals in State")
    plt.title(f"Stochastic SIR models with a population size of {N}")
    plt.savefig("presentation_plot")
    plt.show()
    

    N = 10000

    S_total = N - 1
    I_total = np.ceil(0.05 * N)
    R_total = 0

    t_max = 200
    alpha = 0.4
    beta = 4 / N

    fig = plt.figure(figsize=[6,4],dpi=300)
    fig.tight_layout()
    t, S, I, R = gillespieDirect2Processes(t_max, S_total, I_total,
            R_total, alpha, beta, N)
    S = np.array(S)/100
    I = np.array(I)/100
    R = np.array(R)/100
    plt.plot(t, S, color="#777777",label="Susceptible",lw = 3, figure=fig)
    plt.plot(t, I, color="#c0d7f4",label="Infected",lw = 3, figure=fig)
    plt.plot(t, R, color="#f8b9b3",label="Recovered",lw = 3, figure=fig)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Individuals in State")
    plt.title(f"ODE SIR model with a population size of 100")
    plt.savefig("presentation_plot_ode")
    plt.show()


if __name__=="__main__":
    main()