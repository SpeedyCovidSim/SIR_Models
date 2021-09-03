import numpy as np
from numpy import random
import time
import igraph as ig
import copy
from plots import plotSIR, plotSIRK
from Python_Nonhomog.Epidemic import SIRmax, seirNetworks


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
    numInfNei, numSusNei = initHazards(network, infecteds, N)
    return iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds

def initHazards(network, infecteds, N):
    '''
    inits numInfNei array
    Inputs
    network     : igraph network object
    infecteds   : array of infected vertices
    N           : total number of vertices
    Outputs
    numSusNei   : number of susceptible neighbours for each vertex
    '''
    numInfNei = np.zeros(N)
    numSusNei = np.zeros(N)
    # loop over all infected vertices
    for inf_vert in network.vs(infecteds):
        # Increase number of infected neighbours for neighbouring vertices
        neighbours = network.neighbors(inf_vert)
        for n in neighbours:
            numInfNei[n] += 1
            if not(n in infecteds):
                numSusNei[inf_vert] += 1
    # don't count infecteds for speed-up
    numInfNei[infecteds] = 0
    return numInfNei, numSusNei

def main():
    '''
    Main loop for testing within this Python file
    '''
    # initialise variables
    N = 100
  
    tMax = 200
    maxalpha = 0.4
    maxgamma = 0.1
    maxbeta = 10 / N
    lamb = 0.1
    t = np.linspace(0,tMax,1000)

    # iterate through populations for complete graphs
    print("Beginning toy network simulations")
    Sh = {}
    Eh = {}
    Ih = {}
    Rh = {}
    Sn = {}
    In = {}
    Rn = {}

    def rateFunction(eventType, numSusNei, entry_time,  maxbeta, maxalpha, sim_time): 
        if eventType == "R": 
            return (maxalpha*(sim_time-entry_time))/(25+(sim_time-entry_time)**2)  
        else:
            return (maxbeta*(sim_time-entry_time))/(25+(sim_time-entry_time)**2)


    network = ig.Graph.Erdos_Renyi(100,0.01)
    iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
    for i in range(100):
        print(f"Nonhomog Iteration {i} commencing")
        tp, Sp, Ip, Rp = SIRmax.gillespieMax(tMax, network, iTotal, sTotal, rTotal, numSusNei, susceptible, maxalpha, maxbeta, rateFunction)
        Sn[i] = np.interp(t, tp, Sp, right=Sp[-1])
        In[i] = np.interp(t, tp, Ip, right=Ip[-1])
        Rn[i] = np.interp(t, tp, Rp, right=Rp[-1])
    Snmed = np.median(np.array(list(Sn.values())),0)
    Inmed = np.median(np.array(list(In.values())),0)
    Rnmed = np.median(np.array(list(Rn.values())),0)
    #daily counts for epinow2
    Snmeddaily = np.interp(np.linspace(0,tMax,tMax+1),t,Snmed)
    np.savetxt("Sn daily.csv", Snmeddaily, delimiter=",")

    eTotal = 0
    rates = np.zeros(N)
    # loop over all infected vertices
    for inf_vert in network.vs(infecteds):
        # Increase number of infected neighbours for neighbouring vertices
        neighbours = network.neighbors(inf_vert)
        for n in neighbours:
            rates[n] += maxbeta
    rates[infecteds] = maxalpha

    for i in range(100):
        print(f"Homog Iteration {i} commencing")
        tp, Sp, Ep, Ip, Rp = seirNetworks.gillespieSEIR(tMax, network, eTotal, iTotal, sTotal, rTotal, numInfNei, copy.copy(rates), susceptible, maxalpha, maxbeta, maxgamma)
        Sh[i] = np.interp(t, tp, Sp, right=Sp[-1])
        Eh[i] = np.interp(t, tp, Ep, right=Ep[-1])
        Ih[i] = np.interp(t, tp, Ip, right=Ip[-1])
        Rh[i] = np.interp(t, tp, Rp, right=Rp[-1])
    Shmed = np.median(np.array(list(Sh.values())),0)
    Ehmed = np.median(np.array(list(Eh.values())),0)
    Ihmed = np.median(np.array(list(Ih.values())),0)
    Rhmed = np.median(np.array(list(Rh.values())),0)
    #daily counts for epinow2
    Shmeddaily = np.interp(np.linspace(0,tMax,tMax+1),t,Snmed)
    np.savetxt("Sh daily.csv", Shmeddaily, delimiter=",")

if __name__=="__main__":
    main()