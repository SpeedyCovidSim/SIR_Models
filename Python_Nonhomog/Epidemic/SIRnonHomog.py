import numpy as np
import igraph as ig
from numpy import random

def firstInverseMethod(updateFunction, numProcesses, timeLimit=100):
    '''
    Simulates competing non-Markovian jump processes from a vector of rate functions and max rates
    Using analytical inversion of the distribution function
    Inputs
    updateFunction   : function that returns next event time vector given last event time vector and a vector of uniform variables
    timeLimit        : max simulation time
    Outputs
    eventTimes       : time of events
    '''
    # initialise system
    t = 0
    i = 0
    eventTimes = np.zeros(1000)
    reactionType = np.zeros(1000)
    rng = random.default_rng(123)

    # run till max sim time is reached
    while t < timeLimit:
        # draw next event time and record
        nextEventTimes = updateFunction(t,rng.uniform(size=numProcesses))
        delT = np.min(nextEventTimes)
        reactionType[i] = np.argmin(nextEventTimes)
        t = delT
        eventTimes[i] = t
        # update index
        i += 1

    return eventTimes[0:(i-1)], reactionType[0:(i-1)]

def firstThinningApproxMethod(tMax, network, iTotal, sTotal, rTotal, numInfNei, susceptible, maxalpha, maxbeta, rateFunction, tInit = 0.0):
    '''
    Simulates competing non-Markovian jump processes from a vector of rate functions and max rates
    Using an approximate thinning algorithm on instantaneous rates
    Inputs
    rateFunction      : function that returns vector of rate functions at inputted time
    rateLimit         : max bound on the rate function
    Output
    eventTime    : time of events
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

    # initialise random variate generation with set seed
    rng = random.default_rng(123)

    while t[-1] < tMax and iTotal != 0:
        # individual max bounds
        rateMax = maxbeta*numInfNei*susceptible+maxalpha*(1-susceptible)
        deltaT = rng.exponential(1/rateMax)
        eventIndex = np.argmin(deltaT)
        deltaT = deltaT[eventIndex]
        #thin
        r = rng.uniform()
        if r <= rateFunction(susceptible[eventIndex], numInfNei[eventIndex], entry_times[eventIndex],  maxbeta, maxalpha, sim_time)/rateMax[eventIndex]:
            # update local neighbourhood attributes
            if susceptible[eventIndex]:  # (S->I)
                # change state and entry time
                susceptible[eventIndex] = 0
                entry_times[eventIndex] = t[i-1]+deltaT
                # get neighbouring vertices
                neighbors = network.neighbors(eventIndex)
                # update numfInfNei of neighbours
                for n in neighbors:
                    if susceptible[n]:
                        numInfNei[n] += 1
                # update network totals
                sTotal -= 1
                iTotal += 1

            else: # (I->R)
                # change individual state
                susceptible[eventIndex] = -1
                # get neighbouring vertices
                neighbors = network.neighbors(eventIndex)
                # update numInfNei of neighbours
                for n in neighbors:
                    if susceptible[n]:
                        numInfNei[n] -= 1
                # update network totals
                iTotal -= 1
                rTotal += 1
        # update sim time
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

def nMGA(tMax, network, iTotal, sTotal, rTotal, numInfNei, susceptible, rateFunction, minBounds, tInit = 0.0):
    '''
    Simulates competing non-Markovian jump processes from a passed function of rate functions
    and approximate minimum bounds using the approximate nMGA method
    Inputs
    rateFunction      : function that returns the rate of inputted reaction type at inputted time
    minBounds         : vector of minimum bounds on hazard for finite inter-event calculation
    Output
    eventTime    : time of events
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

    # initialise random variate generation with set seed
    rng = random.default_rng(123)
    instRates = rateFunction(susceptible, network, entry_times, sim_time)
    
    mins = minBounds(susceptible)
    N = np.size(minBounds(susceptible))
    # run till max sim time is reached
    while sim_time < tMax:
        sumRates = np.sum(instRates)
        if abs(0-sumRates)<1e-6:
            mins = minBounds(susceptible)
            minProb = mins/np.sum(mins)
            N = np.size(mins)
            sumRates = np.sum(minBounds)
            deltaT = rng.exponential(1/sumRates)
            eventIndex = random.choice(a=N, p=minProb)
        else:
            deltaT = rng.exponential(1/sumRates)
            eventIndex = random.choice(a=N,p=(instRates/sumRates))
        # update local neighbourhood attributes
        if susceptible[eventIndex]:  # (S->I)
            # change state and entry time
            susceptible[eventIndex] = 0
            entry_times[eventIndex] = t[i-1]+deltaT
            # get neighbouring vertices
            neighbors = network.neighbors(eventIndex)
            # update numfInfNei of neighbours
            for n in neighbors:
                if susceptible[n]:
                    numInfNei[n] += 1
            # update network totals
            sTotal -= 1
            iTotal += 1
        else: # (I->R)
            # change individual state
            susceptible[eventIndex] = -1
            # get neighbouring vertices
            neighbors = network.neighbors(eventIndex)
            # update numInfNei of neighbours
            for n in neighbors:
                if susceptible[n]:
                    numInfNei[n] -= 1
            # update network totals
            iTotal -= 1
            rTotal += 1
        # update rates
        sim_time += deltaT
        instRates = rateFunction(susceptible, network, entry_times, sim_time)

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
        # update index
        i += 1

    # filter totals
    S = S[:i]
    t = t[:i]
    R = R[:i]
    I = I[:i]
    return t, S, I, R

def gillespieMax(tMax, network, iTotal, sTotal, rTotal, numInfNei, susceptible, maxalpha, maxbeta, rateFunction, tInit = 0.0):
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

    # initialise random variate generation with set seed
    rng = random.default_rng(123)
    rates = rateFunction(susceptible, network, entry_times, sim_time, maxbeta, maxalpha)

    while t[-1] < tMax and iTotal != 0:
        # get sum of maximum bounds
        rateMax = maxbeta*numInfNei*susceptible+maxalpha*(1-susceptible)
        H = np.sum(rateMax)
        deltaT = rng.exponential(1/H)
        eventIndex = random.choice(a=N,p=rateMax/H)
        r = rng.uniform()
        if r <= rateFunction(susceptible[eventIndex], numInfNei[eventIndex], entry_times[eventIndex],  maxbeta, maxalpha, sim_time)/rateMax[eventIndex]:
            # update local neighbourhood attributes
            if susceptible[eventIndex]:  # (S->I)
                # change state and entry time
                susceptible[eventIndex] = 0
                entry_times[eventIndex] = t[i-1]+deltaT
                # get neighbouring vertices
                neighbors = network.neighbors(eventIndex)
                # update numfInfNei of neighbours
                for n in neighbors:
                    if susceptible[n]:
                        numInfNei[n] += 1
                # update network totals
                sTotal -= 1
                iTotal += 1

            else: # (I->R)
                # change individual state
                susceptible[eventIndex] = -1
                # get neighbouring vertices
                neighbors = network.neighbors(eventIndex)
                # update numInfNei of neighbours
                for n in neighbors:
                    if susceptible[n]:
                        numInfNei[n] -= 1
                # update network totals
                iTotal -= 1
                rTotal += 1
        # update rates
        sim_time += deltaT
        rates = rateFunction(susceptible, network, entry_times, sim_time, maxbeta, maxalpha)

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