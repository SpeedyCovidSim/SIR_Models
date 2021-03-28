import numpy as np
import igraph as ig
from numpy import random
from plots import plotSIR, plotSIRK

def setNetwork(network, alpha, beta, prop_i=0.05):
    '''
    Initialises a parsed network with required random infected individuals (1 if too small)
    Inputs
    network     : igraph network object
    prop_i      : proportion of population that is infected (default 0.05)
    alpha       : recovery rate
    beta        : infection rate
    Outputs
    iTotal      : number of infected
    sTotal      : number of susceptible
    rTotal      : number of recovered
    numInfNei   : number of infected neighbour of each vertex
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
    numInfNei = initHazards(network, infecteds, N)
    rates = beta*numInfNei
    rates[infecteds] = alpha
    return iTotal, sTotal, rTotal, numInfNei, rates, susceptible

def initHazards(network, infecteds, N):
    '''
    inits numInfNei array
    Inputs
    network     : igraph network object
    infecteds   : array of infected vertices
    N           : total number of vertices
    Outputs
    numInfNei   : number of infected neighbours for each vertex
    '''
    numInfNei = np.zeros(N)
    # loop over all infected vertices
    for inf_vert in network.vs(infecteds):
        # Increase number of infected neighbours for neighbouring vertices
        neighbours = network.neighbors(inf_vert)
        for n in neighbours:
            numInfNei[n] += 1
    # don't count infecteds for speed-up
    numInfNei[infecteds] = 0
    return numInfNei

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


def gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, numInfNei, rates, susceptible, alpha, beta, tInit = 0.0):
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

    # initialise random variate generation with set seed
    rng = random.default_rng(123)
    probs = rates/np.sum(rates)

    while t[-1] < tMax and iTotal != 0:
        # get next event time and next event index
        deltaT, eventIndex = selectEventIndex(rates, probs, rng, N)
        # update local neighbourhood attributes
        if susceptible[eventIndex]:  # (S->I)
            # change state and individual rate
            susceptible[eventIndex] = 0
            rates[eventIndex] = alpha
            # get neighbouring vertices
            neighbors = network.neighbors(eventIndex)
            # update hazards of neighbouring susceptible vertices
            for n in neighbors:
                if susceptible[n]:
                    numInfNei[n] += 1
                    rates[n] = numInfNei[n]*beta
            # update network totals
            sTotal-= 1
            iTotal += 1

        else: # (I->R)
            # change individual rate
            rates[eventIndex] = 0
            # get neighbouring vertices
            neighbors = network.neighbors(eventIndex)
            # update hazards of neighbouring susceptible vertices
            for n in neighbors:
                if susceptible[n]:
                    numInfNei[n] -= 1
                    rates[n] = numInfNei[n]*beta
            # update network totals
            iTotal -= 1
            rTotal += 1

        # update probabilities
        if rates.any():
            probs = rates/np.sum(rates)

        # add totals
        if i < maxI:
            t[i] = t[i-1] + deltaT
            S[i] = sTotal
            I[i] = iTotal
            R[i] = rTotal
        else:
            t.append(t[1] + deltaT)
            S.append(sTotal)
            I.append(iTotal)
            R.append(rTotal)

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
    N = np.array([5, 10, 50, 100,1000,10000])
    k = [2,3,10,20,100,1000]


    tMax = 200
    alpha = 0.4
    beta = 10 / N
    
    # iterate through populations for complete graphs
    if True:
        print("Beginning full graph simulations")
        for i in range(len(N)):
            print(f"Iteration {i} commencing")
            network = ig.Graph.Full(N[i])
            iTotal, sTotal, rTotal, numInfNei, rates, susceptible = setNetwork(network, alpha, beta[i])
            print(f"Beginning simulation {i}")
            t, S, I, R = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, numInfNei, rates, susceptible, alpha, beta[i])
            print(f"Exporting simulation {i}")
            # plot and export the simulation
            outputFileName = f"pythonGraphs/SpeedTest/networkDirectSIR/SIR_Model_Pop_{N[i]}"
            plotSIR(t, [S, I, R], alpha, beta[i], N[i], outputFileName, Display=False)

    if True:   
        print("Beginning connectedness simulations")
        for i in range(len(N)):
            print(f"Iteration {i} commencing")
            network = ig.Graph.K_Regular(N[i], k[i])
            iTotal, sTotal, rTotal, numInfNei, rates, susceptible = setNetwork(network, alpha, beta[i])
            print(f"Beginning simulation {i}")
            t, S, I, R = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, numInfNei, rates, susceptible, alpha, beta[i])
            print(f"Exporting simulation {i}")
            # plot and export the simulation
            outputFileName = f"pythonGraphs/SpeedTest/networkDirectDegreeSIR/SIR_Model_Pop_{N[i]}"
            plotSIRK(t, [S, I, R], alpha, beta[i], N[i], k[i], outputFileName, Display=False)

if __name__=="__main__":
    main()