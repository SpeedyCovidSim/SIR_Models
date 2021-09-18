import numpy as np
from numpy import random
import igraph as ig
import copy
from matplotlib import pyplot as plt
from pythonCompartment.sirNetworksFrequency import gillespieDirectNetwork
from PythonSimulation.simulate import general_SIR_simulation


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

def main(er_small_test=True, ring_small_test=True):
    '''
    Main loop for testing within this Python file
    '''
    print("Beginning small world simulations")
    if (er_small_test):
        # initialise variables
        N = 1000
        tMax = 20
        maxalpha = 0.4
        maxbeta = 4
        j=30
        t = np.linspace(0,tMax,1000)
        print("Beginning Erdos-Renyi small prob. tests")
        # initialise variables
        network = ig.Graph.Full(1000)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"SIR model with a fully connected population size of {N}"
        fname = "PythonPlotting/Small_ER_Tests/Fully_Connected"
        Sfull, Ifull, Rfull = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Erdos_Renyi(1000,0.005)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = f"SIR model with an arc prob. of 0.005, population size of {N}"
        fname = f"PythonPlotting/Small_ER_Tests/005_Connected"
        S5, I5, R5 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Erdos_Renyi(1000,0.003)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = f"SIR model with an arc prob. of 0.003, population size of {N}"
        fname = f"PythonPlotting/Small_ER_Tests/003_Connected"
        S3, I3, R3 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Erdos_Renyi(1000,0.001)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = f"SIR model with an arc prob. of 0.001, population size of {N}"
        fname = f"PythonPlotting/Small_ER_Tests/001_Connected"
        S1, I1, R1 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        fig = plt.figure()
        plt.plot(t, Ifull, color="red",label="Full",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I5, color="blue",label="P = 0.005",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I3, color="green",label="P = 0.003",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I1, color="orange",label="P = 0.001",lw = 2,alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/Small_ER_Tests/P_Comparison")

    if(ring_small_test):
        print("Beginning Small Ring k tests")
        # initialise variables
        N = 1000
        tMax = 20
        maxalpha = 0.4
        maxbeta = 4
        t = np.linspace(0,tMax,1000)
        j=30

        network = ig.Graph.Watts_Strogatz(1,1000,500,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"SIR model with a fully connect ring lattice population {N}"
        fname = f"PythonPlotting/Small_K_Tests/Fully_Connected"
        Sfull, Ifull, Rfull = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Watts_Strogatz(1,1000,40,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title=f"SIR model with a ring lattice population {N}, \n each node connected to nearest 40 neighbours"
        fname=f"PythonPlotting/Small_K_Tests/40_Neighbours"
        S4, I4, R4 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Watts_Strogatz(1,1000,20,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        title=f"SIR model with a ring lattice population {N}, \n each node connected to nearest 20 neighbours"
        fname=f"PythonPlotting/Small_K_Tests/20_Neighbours"
        S2, I2, R2 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Watts_Strogatz(1,1000,10,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title=f"SIR model with a ring lattice population {N}, \n each node connected to nearest 10 neighbours"
        fname=f"PythonPlotting/Small_K_Tests/10_Neighbours"
        S1, I1, R1 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Watts_Strogatz(1,1000,5,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title=f"SIR model with a ring lattice population {N}, \n each node connected to nearest 5 neighbours"
        fname=f"PythonPlotting/Small_K_Tests/5_Neighbours"
        S05, I05, R05 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Watts_Strogatz(1,1000,2,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title=f"SIR model with a ring lattice population {N}, \n each node connected to nearest 2 neighbours"
        fname=f"PythonPlotting/Small_K_Tests/2_Neighbours"
        S02, I02, R02 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        fig = plt.figure()
        plt.plot(t, Ifull, color="red",label="Full",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4, color="blue",label="Neighbourhood = 40",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I2, color="green",label="Neighbourhood = 20",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1, color="yellow",label="Neighbourhood = 10",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I05, color="black",label="Neighbourhood = 5",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I02, color="orange",label="Neighbourhood = 2",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals", fontsize=16)
        plt.title(f"SIR model of ring lattice with varying neighbourhood sizes")
        plt.savefig(f"PythonPlotting/Small_K_Tests/K_Comparison")


if __name__=="__main__":
    main(True, True)
