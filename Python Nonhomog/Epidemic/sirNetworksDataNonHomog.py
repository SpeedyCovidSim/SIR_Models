import numpy as np
import igraph as ig
from numpy import random
from plots import plotSIR, plotSIRK
import time

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