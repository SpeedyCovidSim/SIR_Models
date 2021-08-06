import numpy as np
import igraph as ig
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


def setNetwork(network, prop_i=0.05):
    '''
    Initialises a parsed network with required random infected individuals (1 if too small)
    Inputs
    network     : igraph network object
    prop_i      : proportion of population that is infected (default 0.05)
    Outputs
    iTotal      : number of infected
    sTotal      : number of susceptible
    rTotal      : number of recovered
    numSusNei   : number of susceptible neighbours of each vertex
    rates       : hazard rate of each vertex
    susceptible : boolean array if vertex is susceptible or not
    '''
    # get random proportion of populace to be infected and set states
    N = network.vcount()
    numInfected = int(np.ceil(prop_i*N))
    infecteds = list(random.choice(N, numInfected, replace=False))
    susceptible = np.ones(N)
    susceptible[infecteds] = 0

    # set SIR numbers
    iTotal = numInfected
    sTotal = N-numInfected
    rTotal = 0

    # adding in hazards/rates
    numSusNei = initHazards(network, infecteds, N)
    return iTotal, sTotal, rTotal, numSusNei, susceptible

def initHazards(network, infecteds, N):
    '''
    inits numSusNei array
    Inputs
    network     : igraph network object
    infecteds   : array of infected vertices
    N           : total number of vertices
    Outputs
    numSusNei   : number of susceptible neighbours for each vertex
    '''
    numInfNei = np.zeros(N)
    # loop over all infected vertices
    for inf_vert in network.vs(infecteds):
        # Increase number of susceptible neighbours for neighbouring vertices
        neighbours = network.neighbors(inf_vert)
        for n in neighbours:
            numInfNei[n] += 1
    # don't count infecteds for speed-up
    numInfNei[infecteds] = 0
    return numInfNei

def gillespieNonHomogNetwork(tMax, network, iTotal, sTotal, rTotal, numInfNei, susceptible, maxalpha, maxbeta, rate_function, tInit = 0.0):
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
    rates = rate_function(susceptible, network, entry_times, sim_time, maxbeta, maxalpha)

    while t[-1] < tMax and iTotal != 0:
        # get sum of maximum bounds
        H = S[i-1]*np.max(numInfNei)*maxbeta+I[i-1]*maxalpha
        deltaT = rng.exponential(1/H)
        eventIndex = random.choice(a=N,p=rates/np.sum(rates))
        if (rng.uniform() <= rates[eventIndex]/H):
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
        rates = rate_function(susceptible, network, entry_times, sim_time, maxbeta, maxalpha)

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

def main():
    '''
    Main loop for testing within this Python file
    '''
    # testing the gillespieDirect2Processes function
    # note random seed set within network model so result will occur everytime

    # initialise variables
    # N = np.array([5, 10, 50, 100,1000])
    # k = [2,3,10,20,100]
    N = 100
  
    tMax = 200
    alpha = 0.4
    beta = 10 / N
    
    def rate_function(susceptible, network, entry_times, sim_time, maxbeta, maxalpha):
        temp = np.zeros(len(susceptible))
        rates = np.zeros(len(susceptible))
        for i in range(len(susceptible)):
            if (susceptible[i]==0):
                rates[np.intersect1d(np.nonzero(susceptible==1),network.neighbors(i))] += max(maxbeta*1e-4,maxbeta/10*abs(sim_time-entry_times[i]-5)+maxbeta)
                rates[i] = maxalpha
        return rates


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