import numpy as np
import igraph as ig
from numpy import random
from random import random as unif

def gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, alpha, beta, tInit = 0.0):
    '''
    Direct Gillespie Method, on network
    Uses numpy's random module for r.v. and sampling
    Inputs
    tInit  : Initial time (default of 0)
    tMax   : Simulation end time
    network : population network
    alpha   : probability of infected person recovering [0,1]
    beta    : probability of susceptible person being infected [0,1]
    Outputs
    t       : Array of times at which events have occured
    S       : Array of num people susceptible at each t
    I       : Array of num people infected at each t
    R       : Array of num people recovered at each t
    '''

    # initialise outputs and preallocate space
    N = network.vcount()
    maxI = N*3
    S = np.zeros(maxI)
    t = 1*S
    t[0] = tInit
    I = 1*S
    R = 1*S
    S[0] = sTotal
    I[0] = iTotal
    R[0] = rTotal
    i = 1

    # initialise random variate generation with set seed
    rng = random.default_rng(123)
    probs = rates/np.sum(rates)

    while t[-1] < tMax and iTotal != 0:
        # get next event time and next event index
        deltaT = -np.log(1-unif())/sum(rates)
        eventIndex = random.choice(a=len(rates),p=rates/sum(rates))
        eventType = "I" if eventIndex < N else "R"
        trueIndex = eventIndex if eventIndex < N else(eventIndex-N)
        # update local neighbourhood attributes
        if eventType == "I":  # (S->I)
            # choose infected vertex and update state
            infectedIndex = random.choice(np.intersect1d(np.nonzero(susceptible==1),network.neighbors(trueIndex)))
            susceptible[infectedIndex] = 0
            # update infected vertex's number of susceptible neighbours
            numSusNei[infectedIndex] = len(np.intersect1d(np.nonzero(susceptible==1),network.neighbors(infectedIndex)))
            # update infected vertex's infection and recovery hazards
            rates[N+infectedIndex] = alpha
            rates[infectedIndex] = beta*numSusNei[infectedIndex]/network.degree(infectedIndex)
            # get neighbouring vertices
            neighbors = network.neighbors(infectedIndex)
            # update infection hazards of neighbouring infected vertices
            for n in neighbors:
                if susceptible[n]==0:
                    numSusNei[n] -= 1
                    rates[n] = beta*numSusNei[n]/network.degree(n)
            # update network totals
            sTotal-= 1
            iTotal += 1

        else: # (I->R)
            # change individual recovery and infection hazards and state
            rates[eventIndex] = 0
            rates[trueIndex] = 0
            susceptible[trueIndex] = -1
            # update network totals
            iTotal -= 1
            rTotal += 1

        # add totals
        if i < maxI:
            t[i] = t[i-1] + deltaT
            S[i] = sTotal
            I[i] = iTotal
            R[i] = rTotal
        else:
            t.append(t[-1] + deltaT)
            S.append(sTotal)
            I.append(iTotal)
            R.append(rTotal)

        i += 1
    
    # filter totals
    S = S[:i]
    t = t[:i]
    R = R[:i]
    I = I[:i]
    return t, S, I, R