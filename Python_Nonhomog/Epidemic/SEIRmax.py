import numpy as np
import igraph as ig
from numpy import random

def gillespieMax(tMax, network, eTotal, iTotal, sTotal, rTotal, numSusNei, susceptible, maxalpha, maxbeta, maxgamma, rateFunction, tInit = 0.0):
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
    E = 1*S
    E[0] = eTotal
    S[0] = sTotal
    I[0] = iTotal
    R[0] = rTotal
    i = 1
    sim_time = 0
    entry_times = np.zeros(N)

    # initialise random variate generation with set seed
    rng = random.default_rng(123)

    while t[-1] < tMax and iTotal != 0:
        # get sum of maximum bounds
        rateMax = np.concatenate((maxbeta*numSusNei*(susceptible==0),maxalpha*(susceptible==1),maxgamma*(susceptible==2)))
        H = np.sum(rateMax)
        deltaT = rng.exponential(1/H)
        eventIndex = random.choice(a=N,p=rateMax/H)
        eventType = "E" if eventIndex < N else "R" if eventIndex < 2*N else "I"
        trueIndex = eventIndex if eventIndex < N else (eventIndex-N)
        r = rng.uniform()
        if r <= rateFunction(eventType, numSusNei[trueIndex], entry_times[trueIndex],  maxbeta, maxalpha, maxgamma, sim_time)/rateMax[eventIndex]:
            # update local neighbourhood attributes
            if eventType == "E":  # (S->E)
                # choose infected person
                infectedIndex = random.choice(np.intersect1d(np.nonzero(susceptible==1),network.neighbors(trueIndex)))
                # change state and entry time
                susceptible[infectedIndex] = 2
                entry_times[infectedIndex] = t[i-1]+deltaT
                # get neighbouring vertices
                neighbors = network.neighbors(infectedIndex)
                # update numfSusNei
                for n in neighbors:
                    # if neighbour is susceptible increase own NumSusNei
                    if susceptible[n] == 1:
                        numSusNei[infectedIndex] += 1
                    # if neighbour is infected decrease their NumSusNei
                    elif susceptible[n] == 0:
                        numSusNei[n] -= 1
                # update network totals
                sTotal -= 1
                eTotal += 1
            elif eventType == "I": # (E->I)
                # change state and entry time
                susceptible[trueIndex] = 0
                entry_times[trueIndex] = t[i-1]+deltaT
                eTotal -= 1
                iTotal += 1
            else: # (I->R)
                # change individual state
                susceptible[trueIndex] = -1
                # get neighbouring vertices
                neighbors = network.neighbors(trueIndex)
                # update numSusNei
                numSusNei[trueIndex] = 0
                # update network totals
                iTotal -= 1
                rTotal += 1
        # update rates
        sim_time += deltaT

        # add totals
        if i < maxI:
            t[i] = t[i-1] + deltaT
            S[i] = sTotal
            I[i] = iTotal
            R[i] = rTotal
            E[i] = eTotal
        else:
            t = np.append(t,t[-1]+deltaT)
            S = np.append(S,sTotal)
            I = np.append(I,iTotal)
            R = np.append(R,rTotal)
            E = np.append(E,eTotal)

        i += 1
    
    # filter totals
    S = S[:i]
    t = t[:i]
    R = R[:i]
    I = I[:i]
    E = E[:i]
    return t, S, E, I, R