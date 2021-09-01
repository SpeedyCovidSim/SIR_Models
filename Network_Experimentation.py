import numpy as np
from numpy import random
import time
import igraph as ig
import copy
from matplotlib import pyplot as plt
from pythonCompartment.sirNetworksData import gillespieDirectNetwork


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
    for inf_ind,inf_vert in enumerate(network.vs(infecteds)):
        # Increase number of infected neighbours for neighbouring vertices
        neighbours = network.neighbors(inf_vert)
        for n in neighbours:
            numInfNei[n] += 1
            if not(n in infecteds):
                numSusNei[infecteds[inf_ind]] += 1
    # don't count infecteds for speed-up
    numInfNei[infecteds] = 0
    return numInfNei, numSusNei

def main():
    '''
    Main loop for testing within this Python file
    '''
    # initialise variables
    N = 1000
  
    tMax = 200
    maxalpha = 0.4
    maxgamma = 0.1
    maxbeta = 4
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

    network = ig.Graph.Full(1000)
    iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
    rates = np.zeros(2*N)
    for inf in infecteds:
        # set recovery hazard
        rates[N+inf] = maxalpha
        # set infection hazard
        rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
    
    tp, Sp, Ip, Rp = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta)
    fig = plt.figure()
    plt.plot(tp, Sp, color="#82c7a5",label="Susceptible",lw = 2, alpha=0.5,figure=fig)
    plt.plot(tp, Ip, color="#f15e22",label="Infected",lw = 2, alpha=0.5,figure=fig)
    plt.plot(tp, Rp, color="#7890cd",label="Recovered",lw = 2, alpha=0.5,figure=fig)
    plt.legend()
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Number of Individuals in State", fontsize=16)
    plt.title(f"SIR model with a fully connected population size of {N}", fontsize=20)
    plt.show()

    # for i in range(100):
    #     print(f"Nonhomog Iteration {i} commencing")
    #     tp, Sp, Ip, Rp = SIRmax.gillespieMax(tMax, network, iTotal, sTotal, rTotal, numSusNei, susceptible, maxalpha, maxbeta, rateFunction)
    #     Sn[i] = np.interp(t, tp, Sp, right=Sp[-1])
    #     In[i] = np.interp(t, tp, Ip, right=Ip[-1])
    #     Rn[i] = np.interp(t, tp, Rp, right=Rp[-1])
    # Snmed = np.median(np.array(list(Sn.values())),0)
    # Inmed = np.median(np.array(list(In.values())),0)
    # Rnmed = np.median(np.array(list(Rn.values())),0)
    # #daily counts for epinow2
    # Snmeddaily = np.interp(np.linspace(0,tMax,tMax+1),t,Snmed)
    # np.savetxt("Sn daily.csv", Snmeddaily, delimiter=",")

    # for i in range(100):
    #     print(f"Homog Iteration {i} commencing")
    #     tp, Sp, Ep, Ip, Rp = seirNetworks.gillespieSEIR(tMax, network, eTotal, iTotal, sTotal, rTotal, numInfNei, copy.copy(rates), susceptible, maxalpha, maxbeta, maxgamma)
    #     Sh[i] = np.interp(t, tp, Sp, right=Sp[-1])
    #     Eh[i] = np.interp(t, tp, Ep, right=Ep[-1])
    #     Ih[i] = np.interp(t, tp, Ip, right=Ip[-1])
    #     Rh[i] = np.interp(t, tp, Rp, right=Rp[-1])
    # Shmed = np.median(np.array(list(Sh.values())),0)
    # Ehmed = np.median(np.array(list(Eh.values())),0)
    # Ihmed = np.median(np.array(list(Ih.values())),0)
    # Rhmed = np.median(np.array(list(Rh.values())),0)
    # #daily counts for epinow2
    # Shmeddaily = np.interp(np.linspace(0,tMax,tMax+1),t,Snmed)
    # np.savetxt("Sh daily.csv", Shmeddaily, delimiter=",")

if __name__=="__main__":
    main()
