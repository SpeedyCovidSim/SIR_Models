import numpy as np
from numpy import random
import time
from matplotlib import pyplot as plt

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
    rng = random.default_rng()

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

def firstThinningApproxMethod(rateFunction, rateMax, timeLimit=100):
    '''
    Simulates competing non-Markovian jump processes from a vector of rate functions and max rates
    Using an approximate thinning algorithm on instantaneous rates
    Inputs
    rateFunction      : function that returns vector of rate functions at inputted time
    rateLimit         : max bound on the rate function
    Output
    eventTime    : time of events
    '''
    # initialise system
    t = 0
    i = 0
    eventTimes = np.zeros(1000)
    reactionType = np.zeros(1000)
    rng = random.default_rng()
    # run till max sim time is reached
    while t < timeLimit:
        #draw inter-event times
        delTs = rng.exponential(1/rateMax)
        reactType = np.argmin(delTs)
        delT = np.min(delTs)
        #thin
        r = rng.uniform()
        if r <= rateFunction(reactType, t)/rateMax[reactType]:
            reactionType[i] = reactType
            eventTimes[i] = t + delT
            # update index
            i += 1
        # update sim time
        t += delT

    return eventTimes[0:(i-1)], reactionType[0:(i-1)]



def gillespieMax(rateFunction, rateMax, timeLimit=100):
    '''
    Simulates competing non-Markovian jump processes from a passed function of rate functions and max rates
    Using a thinned Gillespie Direct algorithm
    Inputs
    rateFunction      : function that returns the rate of inputted reaction type at inputted time
    rateLimit         : max bound on the rate function
    Output
    eventTime    : time of events
    '''
    # initialise system
    t = 0
    i = 0
    eventTimes = np.zeros(1000)
    reactionType = np.zeros(1000)
    rng = random.default_rng()
    sumH = np.sum(rateMax)
    N = np.size(rateMax)
    probs = rateMax/sumH
    # run till max sim time is reached
    while t < timeLimit:
        delT = rng.exponential(1/sumH)
        reactType = random.choice(a=N,p=probs)
        r = rng.uniform()
        if r <= rateFunction(reactType, t)/rateMax[reactType]:
            reactionType[i] = reactType
            eventTimes[i] = t + delT
            # update index
            i += 1
        t += delT

    return eventTimes[0:(i-1)], reactionType[0:(i-1)]

def nMGA(rateFunction, minBounds, timeLimit=100):
    '''
    Simulates competing non-Markovian jump processes from a passed function of rate functions
    and approximate minimum bounds using the approximate nMGA method
    Inputs
    rateFunction      : function that returns the rate of inputted reaction type at inputted time
    minBounds         : vector of minimum bounds on hazard for finite inter-event calculation
    Output
    eventTime    : time of events
    '''
    # initialise system
    t = 0
    i = 0
    eventTimes = np.zeros(1000)
    reactionType = np.zeros(1000)
    rng = random.default_rng()
    N = np.size(minBounds)
    minProb = minBounds/np.sum(minBounds)
    # run till max sim time is reached
    while t < timeLimit:
        instRates = rateFunction(t)
        sumRates = np.sum(instRates)
        if abs(0-sumRates)<1e-6:
            sumRates = np.sum(minBounds)
            delT = rng.exponential(1/sumRates)
            reactType = random.choice(a=N, p=minProb)
            reactionType[i] = reactType
            t += delT
            eventTimes[i] = t
        else:
            delT = rng.exponential(1/sumRates)
            reactType = random.choice(a=N,p=(instRates/sumRates))
            reactionType[i] = reactType
            t += delT
            eventTimes[i] = t
        # update index
        i += 1

    return eventTimes[0:(i-1)], reactionType[0:(i-1)]



def main():
    '''
    Main loop for testing within this Python file
    '''
    # simSizes = [1,10,100,1000,10000]
    # times = {}
    # times['firstInverse'] = np.zeros(len(simSizes))
    # times['firstThin'] = np.zeros(len(simSizes))
    # times['gillespieMax'] = np.zeros(len(simSizes))
    # times['nMGA'] = np.zeros(len(simSizes))
    # for i in range(len(simSizes)):
    #     start = time.time()
    #     for i in range(10):
    #         invEventTimes, invReactTypes = firstInverseMethod(updateFunction, numProcesses, timeLimit=40)
    #     end = time.time()
    #     times['firstInverse'][i]
    
    # for i in range(len(simSizes)):
    #     for i in range(10):
    #         start = time.time()
    #         firstMaxEventTimes, firstMaxReactTypes = firstThinningApproxMethod(rateFunctionSing, rateMax, timeLimit=40)
    #         end = time.time()

    # for i in range(len(simSizes)):
    #     for i in range(10):
    #         start = time.time()
    #         MaxEventTimes, MaxReactTypes = gillespieMax(rateFunctionSing, rateMax, timeLimit=40)
    #         end = time.time()

    # for i in range(len(simSizes)):
    #     for i in range(10):
    #         start = time.time()
    #         nMGAEventTimes, nMGAReactTypes = nMGA(rateFunctionVect, minBounds, timeLimit=40)
    #         end = time.time()
    
    print("timing done")
    


if __name__=="__main__":
    main()