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
    rng = random.default_rng(123)
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
    rng = random.default_rng(123)
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
    rng = random.default_rng(123)
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
    # testing the output of the two simulation functions
    # note random seed set within simulation functions to standardise output
  
    timeLimit = 100
    
    def updateFunction(ti, u):
        tip1 = [
            np.sqrt((1+ti**2-u**(2/4))/(u**(2/4))), 
            np.sqrt((25+ti**2-25*u**(2/20))/(u**(2/20))),
            np.sqrt((81+ti**2-81*u**(2/40))/(u**(2/40))), 
            np.sqrt((169+ti**2-169*u**(2/60))/(u**(2/60)))]
        return tip1

    def rateFunctionSing(reactType, t): 
        if reactType == 0: 
            return (4*t)/(1+t**2) 
        elif reactType == 1: 
            return (20*t)/(25+t**2) 
        elif reactType == 2: 
            return (40*t)/(81+t**2) 
        else:
            return (60*t)/(169+t**2)

    def rateFunctionVect(t):
        rates = [(4*t)/(1+t**2),(20*t)/(25+t**2),(40*t)/(81+t**2),(60*t)/(169+t**2)]
        return rates

    # system params
    numProcesses = 4
    rateMax = np.array([2,2,2.23,2.31])
    minBounds = np.array([0.4,0.2,0.1,0.1])

    # run each simulation type
    invEventTimes, invReactTypes = firstInverseMethod(updateFunction, numProcesses, timeLimit=40)
    firstMaxEventTimes, firstMaxReactTypes = firstThinningApproxMethod(rateFunctionSing, rateMax, timeLimit=40)
    MaxEventTimes, MaxReactTypes = gillespieMax(rateFunctionSing, rateMax, timeLimit=40)
    nMGAEventTimes, nMGAReactTypes = nMGA(rateFunctionVect, minBounds, timeLimit=40)
    
    # get event counts and plot each simulation
    inv0 = np.cumsum(invReactTypes==0)
    inv1 = np.cumsum(invReactTypes==1)
    inv2 = np.cumsum(invReactTypes==2)
    inv3 = np.cumsum(invReactTypes==3)

    fig = plt.figure()
    plt.plot(invEventTimes, inv0, label="Event Type 0",color="#82c7a5",lw = 2, figure=fig)
    plt.plot(invEventTimes, inv1, label="Event Type 1",color="#f15e22",lw = 2, figure=fig)
    plt.plot(invEventTimes, inv2, label="Event Type 2",color="#7890cd",lw = 2, figure=fig)
    plt.plot(invEventTimes, inv3, label="Event Type 3",color="#ffd966",lw = 2, figure=fig)
    plt.legend(fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Cum. Number of Events", fontsize=16)
    plt.title(f"Inv. Simulation of 4 Competing Poisson Processes", fontsize=20)
    plt.show()


    first0 = np.cumsum(firstMaxReactTypes==0)
    first1 = np.cumsum(firstMaxReactTypes==1)
    first2 = np.cumsum(firstMaxReactTypes==2)
    first3 = np.cumsum(firstMaxReactTypes==3)

    fig = plt.figure()
    plt.plot(firstMaxEventTimes, first0, label="Event Type 0",color="#82c7a5",lw = 2, figure=fig)
    plt.plot(firstMaxEventTimes, first1, label="Event Type 1",color="#f15e22",lw = 2, figure=fig)
    plt.plot(firstMaxEventTimes, first2, label="Event Type 2",color="#7890cd",lw = 2, figure=fig)
    plt.plot(firstMaxEventTimes, first3, label="Event Type 3",color="#ffd966",lw = 2, figure=fig)
    plt.legend(fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Cum. Number of Events", fontsize=16)
    plt.title(f"FirstMax Simulation of 4 Competing Poisson Processes", fontsize=20)
    plt.show()


    max0 = np.cumsum(MaxReactTypes==0)
    max1 = np.cumsum(MaxReactTypes==1)
    max2 = np.cumsum(MaxReactTypes==2)
    max3 = np.cumsum(MaxReactTypes==3)

    fig = plt.figure()
    plt.plot(MaxEventTimes, max0, label="Event Type 0",color="#82c7a5",lw = 2, figure=fig)
    plt.plot(MaxEventTimes, max1, label="Event Type 1",color="#f15e22",lw = 2, figure=fig)
    plt.plot(MaxEventTimes, max2, label="Event Type 2",color="#7890cd",lw = 2, figure=fig)
    plt.plot(MaxEventTimes, max3, label="Event Type 3",color="#ffd966",lw = 2, figure=fig)
    plt.legend(fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Cum. Number of Events", fontsize=16)
    plt.title(f"Gillespie Max Simulation of 4 Competing Poisson Processes", fontsize=20)
    plt.show()


    nMGA0 = np.cumsum(nMGAReactTypes==0)
    nMGA1 = np.cumsum(nMGAReactTypes==1)
    nMGA2 = np.cumsum(nMGAReactTypes==2)
    nMGA3 = np.cumsum(nMGAReactTypes==3)

    fig = plt.figure()
    plt.plot(nMGAEventTimes, nMGA0, label="Event Type 0",color="#82c7a5",lw = 2, figure=fig)
    plt.plot(nMGAEventTimes, nMGA1, label="Event Type 1",color="#f15e22",lw = 2, figure=fig)
    plt.plot(nMGAEventTimes, nMGA2, label="Event Type 2",color="#7890cd",lw = 2, figure=fig)
    plt.plot(nMGAEventTimes, nMGA3, label="Event Type 3",color="#ffd966",lw = 2, figure=fig)
    plt.legend(fontsize=20)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Cum. Number of Events", fontsize=16)
    plt.title(f"nMGA Simulation of 4 Competing Poisson Processes", fontsize=20)
    plt.show()


    print('done')

if __name__=="__main__":
    main()