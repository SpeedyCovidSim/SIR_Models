import numpy as np
from numpy import random
from plots import plotSIR, plotSIRK
import time

def inverseMethod(updateFunction, timeLimit=100):
    '''
    Simulates a nonhomogeneous Poisson Process from a given time update function
    Using analytical inversion of the distribution function
    Inputs
    updateFunction   : function that returns next event time given last event time and a uniform variable
    timeLimit        : max simulation time
    Outputs
    eventTimes       : time of events
    '''
    # initialise system
    t = 0
    i = 0
    eventTimes = np.zeros(1000)
    rng = random.default_rng(123)

    # run till max sim time is reached
    while t < timeLimit:
        # draw next event time and record
        t = updateFunction(t,rng.uniform())
        eventTimes[i] = t
        # update index
        i += 1

    return eventTimes

def thinningExactMethod(rateFunction, rateMax, timeLimit=100):
    '''
    Simulates a nonhomogeneous Poisson Process from a given rate function and max rate
    Using an exact thinning algorithm on future rates
    Inputs
    rateFunction : rate function of the nonhomogeneous process
    rateMax      : max bound on the rate function
    Output
    eventTime    : time of events
    '''
    # initialise system
    t = 0
    i = 0
    eventTimes = np.zeros(1000)
    rng = random.default_rng(123)
    # run till max sim time is reached
    while t < timeLimit:
        #draw inter-event time
        delT = rng.exponential(1/rateMax)
        #update simulation time
        t += delT
        # thin process
        u = rng.uniform()
        if u <= (rateFunction(t)/rateMax):
            eventTimes[i] = t
        # update index
        i += 1

    return eventTimes

def thinningApproxMethod(rateFunction, rateMax, timeLimit=100):
    '''
    Simulates a nonhomogeneous Poisson Process from a given rate function and max rate
    Using an approx thinning algorithm on instantaneous rates
    Inputs
    rateFunction : rate function of the nonhomogeneous process
    rateMax      : max bound on the rate function
    Output
    eventTime    : time of events
    '''
    # initialise system
    t = 0
    i = 0
    eventTimes = np.zeros(1000)
    rng = random.default_rng(123)
    # run till max sim time is reached
    while t < timeLimit:
        #draw inter-event time
        delT = rng.exponential(1/rateMax)
        # thin process
        u = rng.uniform()
        if u <= (rateFunction(t)/rateMax):
            eventTimes[i] = t
        # update index and time
        t += delT
        i += 1

    return eventTimes


def main():
    '''
    Main loop for testing within this Python file
    '''
    # testing the output of the two simulation functions
    # note random seed set within simulation functions to standardise output

    N = 100
  
    timeLimit = 100
    
    def rate_function(ti, u):
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