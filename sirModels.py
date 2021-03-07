'''
This is a code base for a simple SIR simulation using the Gillespie Direct
Method. Contains the simulation function.

Author: Joel Trent and Josh Looker
'''
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
    beta    : probability of susceptible person being infected [0,1]

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
