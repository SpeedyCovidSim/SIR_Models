import numpy as np
import igraph as ig
from numpy import random
from plots import plotSIR, plotSIRK

def selectEventIndex(rates, probs, rng, N):
    '''
    finds time and index of next event
    Inputs
    rates       : array of hazard rates
    probs       : array of hazard probabilities
    rng         : rng object
    N           : total number of vertices
    Outputs
    deltaT      : time till next event
    eventIndex  : index of next event
    '''
    # get hazard sum and next event time
    h = np.sum(rates)
    deltaT = rng.exponential(1/h)
    # choose the index of the individual to transition
    eventIndex = random.choice(a=N,p=probs)
    return deltaT, eventIndex


def gillespieSEIR(tMax, network, eTotal, iTotal, sTotal, rTotal, numInfNei, rates, susceptible, alpha, beta, gamma, tInit = 0.0):
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

    # initialise random variate generation with set seed
    rng = random.default_rng(123)
    probs = rates/np.sum(rates)

    while t[-1] < tMax and iTotal != 0:
        # get next event time and next event index
        deltaT, eventIndex = selectEventIndex(rates, probs, rng, N)
        # update local neighbourhood attributes
        if susceptible[eventIndex]:  # (S->E)
            # change state and individual rate
            susceptible[eventIndex] = -1
            rates[eventIndex] = gamma
            # update network totals
            sTotal -= 1
            eTotal += 1

        elif susceptible[eventIndex] == -1: # (E->I)
            # change state and individual rate
            susceptible[eventIndex] = 0
            rates[eventIndex] = alpha
            # get neighbouring vertices
            neighbors = network.neighbors(eventIndex)
            # update hazards of neighbouring susceptible vertices
            for n in neighbors:
                if susceptible[n] == 1:
                    numInfNei[n] -= 1
                    rates[n] = numInfNei[n]*beta
            # update network totals
            eTotal -= 1
            iTotal += 1

        else: # (I->R)
            # change individual rate
            rates[eventIndex] = 0
            # get neighbouring vertices
            neighbors = network.neighbors(eventIndex)
            # update hazards of neighbouring susceptible vertices
            for n in neighbors:
                if susceptible[n] == 1:
                    numInfNei[n] -= 1
                    rates[n] = numInfNei[n]*beta
            # update network totals
            iTotal -= 1
            rTotal += 1

        # update probabilities
        probs = rates/np.sum(rates)

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