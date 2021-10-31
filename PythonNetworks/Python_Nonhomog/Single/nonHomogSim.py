import numpy as np
from numpy import random
from random import random as unif
from plots import plotSIR, plotSIRK
import time

def inverseMethod(updateFunction, timeLimit=50):
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
    numEvents = np.zeros(1000)

    # run till max sim time is reached
    while t < timeLimit:
        # draw next event time and record
        t = updateFunction(t,unif())
        eventTimes[i] = t
        numEvents[i] = numEvents[i-1] + 1
        # update index
        i += 1
    return eventTimes[:i], numEvents[:i]

def thinningExactMethod(rateFunction, rateMax, timeLimit=50):
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
    numEvents = np.zeros(1000)
    rng = random.default_rng()
    # run till max sim time is reached
    while t < timeLimit:
        #draw inter-event time
        delT = rng.exponential(1/rateMax)
        #update simulation time
        t += delT
        # thin process
        u = unif()
        if u <= (rateFunction(t)/rateMax):
            eventTimes[i+1] = t
            numEvents[i] = numEvents[i-1] + 1
            # update index
            i += 1
    return eventTimes[:i], numEvents[:i]

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
    numEvents = np.zeros(1000)
    rng = random.default_rng()
    # run till max sim time is reached
    while t < timeLimit:
        #draw inter-event time
        delT = rng.exponential(1/rateMax)
        # thin process
        u = unif()
        if u <= (rateFunction(t)/rateMax):
            eventTimes[i] = t
            numEvents[i] = numEvents[i-1] + 1
            # update index
            i += 1
        # update time
        t += delT
    return eventTimes[:i], numEvents[:i]
