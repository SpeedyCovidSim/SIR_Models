import numpy as np
from numpy import random
from plots import plotSIR, plotSIRK
import time

def inverseMethod(updateFunction, timeLimit=100):
    '''
    Simulates a nonhomogeneous Poisson Process from a given time update function
    Inputs
    updateFunction   : function that returns next event time (ti+1) given last even time (ti)
    timeLimit        : max simulation time
    Outputs
    eventTimes       : time of events
    '''
    # initialise system
    t = 0
    i = 0
    eventTimes = np.zeros(1000)

    # run till max sim time is reached
    while t < timeLimit:
        # draw next event time and record
        t = updateFunction(t)
        eventTimes[i] = t
    return eventTimes


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