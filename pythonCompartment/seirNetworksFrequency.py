import numpy as np
import igraph as ig
from numpy import random
from random import random as unif

def gillespieSEIR(tMax, network, eTotal, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, alpha, beta, gamma, tInit = 0.0):
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
    E       : Array of num people exposed at each t
    I       : Array of num people infected at each t
    R       : Array of num people recovered at each t
    '''

    # initialise outputs and preallocate space
    N = network.vcount()
    maxI = N*3
    S = np.zeros(maxI)
    t = 1*S
    t[0] = tInit
    E = 1*S
    I = 1*S
    R = 1*S
    E[0] = eTotal
    S[0] = sTotal
    I[0] = iTotal
    R[0] = rTotal
    i = 1

    while t[-1] < tMax and iTotal != 0:
        # get next event time and next event index
        deltaT = -np.log(1-unif())/sum(rates)
        eventIndex = random.choice(a=len(rates),p=rates/sum(rates))
        eventType = "E" if eventIndex < N else "R" if eventIndex < 2*N else "I"
        trueIndex = eventIndex if eventIndex<N else (eventIndex-N) if eventIndex <(2*N) else (eventIndex-2*N)
        # update local neighbourhood attributes
        if eventType == "E":  # (S->E)
            # choose infected person
            infectedIndex = random.choice(np.intersect1d(np.nonzero(susceptible==1),network.neighbors(trueIndex)))
            # change state
            susceptible[infectedIndex] = 2
            # update infected vertex's number of susceptible neighbours
            numSusNei[infectedIndex] = len(np.intersect1d(np.nonzero(susceptible==1),network.neighbors(infectedIndex)))
            # get neighbouring vertices
            neighbors = network.neighbors(infectedIndex)
            # update infection hazards of neighbouring infected vertices
            for n in neighbors:
                # if neighbour is infected decrease their NumSuNei and update hazard
                if (susceptible[n]==0 or susceptible[n]==2):
                    numSusNei[n] -= 1
                    if susceptible[n]==0:
                        rates[n] = beta*numSusNei[n]/network.degree(n) if network.degree(n)>0 else 0
            # update exposed person's disease progression hazard
            rates[infectedIndex+2*N] = gamma
            # update network totals
            sTotal -= 1
            eTotal += 1

        elif eventType == "I": # (E->I)
            # change state and update rates (infection and recovery)
            susceptible[trueIndex] = 0
            rates[eventIndex] = 0
            rates[trueIndex+N] = alpha
            rates[trueIndex] = beta*numSusNei[trueIndex]/network.degree(trueIndex) if network.degree(trueIndex)>0 else 0
            eTotal -= 1
            iTotal += 1

        else: # (I->R)
            # change individual rate
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
            E[i] = eTotal
            I[i] = iTotal
            R[i] = rTotal
        else:
            t.append(t[-1] + deltaT)
            S.append(sTotal)
            E.append(eTotal)
            I.append(iTotal)
            R.append(rTotal)

        i += 1
    
    # filter totals
    S = S[:i]
    E = E[:i]
    t = t[:i]
    R = R[:i]
    I = I[:i]
    return t, S, E, I, R