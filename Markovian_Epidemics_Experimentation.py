import igraph as ig
import numpy as np
from pythonCompartment.sirNetworksDensity import gillespieDirectNetwork as g_func
from pythonCompartment.sirFirstReact import gillespieFirstNetwork as f_func
from PythonSimulation.simulate import ode_N_simulation
from numpy import random

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
    # initialise variables
    Ns = [100,100]
    maxbeta = 4
    maxalpha = 0.4
    tMax = 40

    network = ig.Graph.Full(Ns[0])
    network1 = ig.Graph.Full(Ns[1])
    iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
    iTotal1, sTotal1, rTotal1, numInfNei, numSusNei1, susceptible1, infecteds1 = setNetwork(network1)
    rates = np.zeros(2*Ns[0])
    rates1 = np.zeros(2*Ns[1])
    for inf in infecteds:
            # set recovery hazard
            rates[Ns[0]+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)

    for inf in infecteds1:
        # set recovery hazard
        rates1[Ns[1]+inf] = maxalpha
        # set infection hazard
        rates1[inf] = maxbeta*numSusNei1[inf]/network1.degree(inf)
    title = f"Stochastic SIR models with varying population sizes"
    fname = "PythonPlotting/Comparisons/ODE"
    ode_N_simulation(g_func, f_func, tMax, [network, network1], [iTotal,iTotal1], [sTotal,sTotal1], [rTotal,rTotal1], [numSusNei,numSusNei1], [rates,rates1], [susceptible,susceptible1], maxalpha, maxbeta, title, fname)


if __name__=="__main__":
    main()