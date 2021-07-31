import numpy as np
from numpy import random
from plots import plotSIR, plotSIRK
import time

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
        t += delT
        eventTimes[i] = t
        # update index
        i += 1

    return eventTimes, reactionType

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
        # update sim time
        t += delT
        # update index
        i += 1

    return eventTimes, reactionType



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
        t += delT
        # update index
        i += 1

    return eventTimes, reactionType

def nMGA():
    pass



def main():
    '''
    Main loop for testing within this Python file
    '''
    # testing the output of the two simulation functions
    # note random seed set within simulation functions to standardise output

    N = 100
  
    timeLimit = 100
    
    def updateFunction(ti, u):
        tip1 = np.sqrt((ti^2+25*u)/(1-u))
        return tip1


    # iterate through populations for complete graphs
    if True:
        print("Beginning Ernos-Renyi simulations")
        start = time.time()

        for i in range(1):
            print(f"Iteration {i} commencing")
            network = ig.Graph.Erdos_Renyi(100,0.1)
            iTotal, sTotal, rTotal, numInfNei, susceptible = setNetwork(network)
            print(f"Beginning simulation {i}")
            t, S, I, R = gillespieNonHomogNetwork(tMax, network, iTotal, sTotal, rTotal, numInfNei, susceptible, alpha, beta, rate_function)
        end = time.time()
        print(f"Avg. time taken for Lumped hazards simulation: {(end-start)/10}")

        print(f"Exporting last simulation {i}")
        # plot and export the simulation
        outputFileName = f"pythonGraphs/nonMarkovSim/SIR_Model_Pop_{N}"
        plotSIR(t, [S, I, R], alpha, beta, N, outputFileName, Display=False)

if __name__=="__main__":
    main()