import numpy as np
import igraph as ig
from numpy import random
from random import random as unif

def gillespieMax(tMax, network, iTotal, sTotal, rTotal, numSusNei, susceptible, rateParams, rateFunction, rateMax, tInit = 0.0):
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
    num_neighbours = np.array(network.degree())
    maxalpha = rateMax[0]
    maxbeta = rateMax[1]
    kinf, laminf, krec, lamrec = rateParams
    
    while t[-1] < tMax and iTotal != 0:
        # get sum of maximum bounds
        rateMax = np.concatenate((maxbeta*numSusNei/num_neighbours,maxalpha*(1-np.abs(susceptible))))
        rateMax[np.isnan(rateMax)] = 0
        H = np.sum(rateMax)
        deltaT = -np.log(1-unif())/H
        eventIndex = random.choice(a=len(rateMax),p=rateMax/H)
        eventType = "I" if eventIndex < N else "R"
        trueIndex = eventIndex if eventIndex < N else (eventIndex-N)
        r = unif()
        if r <= rateFunction(eventType, numSusNei[trueIndex], num_neighbours[trueIndex], entry_times[trueIndex], kinf, laminf, krec, lamrec, sim_time)/rateMax[eventIndex]:
            # update local neighbourhood attributes
            if eventType == "I":  # (S->I)
                # choose infected person
                infectedIndex = random.choice(np.intersect1d(np.nonzero(susceptible==1),network.neighbors(trueIndex)))
                # change state and entry time
                susceptible[infectedIndex] = 0
                entry_times[infectedIndex] = t[i-1]+deltaT
                # update infected vertex's number of susceptible neighbours
                numSusNei[infectedIndex] = len(np.intersect1d(np.nonzero(susceptible==1),network.neighbors(infectedIndex)))
                # get neighbouring vertices
                neighbors = network.neighbors(infectedIndex)
                # update numSusNei of neighbouring infected vertices
                for n in neighbors:
                    if susceptible[n] == 0:
                        numSusNei[n] -= 1
                # update network totals
                sTotal -= 1
                iTotal += 1

            else: # (I->R)
                # change individual state
                susceptible[trueIndex] = -1
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