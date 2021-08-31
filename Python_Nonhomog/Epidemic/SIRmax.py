import numpy as np
import igraph as ig
from numpy import random

def gillespieMax(tMax, network, iTotal, sTotal, rTotal, numSusNei, susceptible, maxalpha, maxbeta, rateFunction, tInit = 0.0):
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
    sim_time = 0
    entry_times = np.zeros(N)

    while t[-1] < tMax and iTotal != 0:
        # get sum of maximum bounds
        rateMax = np.concatenate((maxbeta*numSusNei,maxalpha*(1-np.abs(susceptible))))
        H = np.sum(rateMax)
        deltaT = -np.log(1-random())/H
        eventIndex = random.choice(a=N,p=rateMax/H)
        eventType = "I" if eventIndex < N else "R"
        trueIndex = eventIndex if eventIndex < N else (eventIndex-N)
        r = random.uniform()
        if r <= rateFunction(eventType, numSusNei[trueIndex], entry_times[trueIndex],  maxbeta, maxalpha, sim_time)/rateMax[eventIndex]:
            # update local neighbourhood attributes
            if eventType == "I":  # (S->I)
                # choose infected person
                infectedIndex = random.choice(np.intersect1d(np.nonzero(susceptible==1),network.neighbors(trueIndex)))
                # change state and entry time
                susceptible[infectedIndex] = 0
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
        else:
            t = np.append(t,t[-1]+deltaT)
            S = np.append(S,sTotal)
            I = np.append(I,iTotal)
            R = np.append(R,rTotal)

        i += 1
    
    # filter totals
    S = S[:i]
    t = t[:i]
    R = R[:i]
    I = I[:i]
    return t, S, I, R