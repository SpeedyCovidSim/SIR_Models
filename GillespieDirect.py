'''
This is a code base for a simple SIR simulation using the Gillespie Direct
Method

Author: Joel Trent and Josh Looker
'''
from matplotlib import pyplot as plt
import numpy as np
from random import random

def gillespieDirect2Processes(t_max, S_total, I_total, R_total, alpha, beta, N, t_init=0):
    '''
    Inputs
    t_init  : Initial time (default of 0)
    t_max   : Simulation end time
    S_total : Num people susceptible to infection
    I_total : Num people infected
    R_total : Num people recovered
    N       : Population size
    alpha   : probability of infected person recovering [0,1]
    gamma   : probability of susceptible person being infected [0,1]

    Outputs
    t       : Array of times at which events have occured
    S       : Array of Num people susceptible at each t
    I       : Array of Num people infected at each t
    R       : Array of Num people recovered at each t
    '''


    # initialise outputs
    t = [t_init]
    S = [S_total]
    I = [I_total]
    R = [R_total]

    while t[-1] < t_max and I_total != 0:
        # calculate the propensities to transition
        # h1 is propensity for infection, h2 is propensity for recovery
        h_i = np.array([beta * I_total * S_total, alpha * I_total])
        h = sum(h_i)

        # time to any event occurring
        delta_t = -np.log(1-random())/h
        #println(delta_t)

        # selection probabilities for each transition process. sum(j) = 1
        j = h_i / h

        # coding this way so can include more processes later with additional
        # elseif
        # could be done more efficiently if large number of processes
        choice = random()

        if choice < j[0] and S_total != 0:  # (S->I)
            S_total -= 1
            I_total += 1
        else:    # (I->R) (assumes that I is not 0)
            I_total -= 1
            R_total += 1

        t.append(t[-1]+delta_t)
        S.append(S_total)
        I.append(I_total)
        R.append(R_total)

    return t, S, I , R


def plots(t, SIR, N, Display=True, save=True):
    '''
    Inputs
    t       : Array of times at which events have occured
    SIR     : Array of arrays of Num people susceptible, infected and recovered at
                each t
    N       : Population size

    Outputs
    png     : plot of SIR model over time [by default]
    '''
    fig = plt.figure()
    plt.plot(t, SIR[0], label="Susceptible", lw = 2, figure=fig)
    plt.plot(t, SIR[1], label="Infected", lw = 2, figure=fig)
    plt.plot(t, SIR[2], label="Recovered", lw=2, figure=fig)

    plt.xlabel("Time")
    plt.ylabel("Population Number")
    plt.title("SIR model over time with a population size of $N")
    plt.legend()

    if Display:
        # required to display graph on plots.
        fig.show()

    if save:
        # Save graph as pngW
        fig.savefig(f"pythonGraphs/SIR_Model_Pop_{N}")

#-----------------------------------------------------------------------------
# testing the gillespieDirect2Processes function

# Get same thing each time


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

    plots(t, [S, I, R], N[i],Display=False)
