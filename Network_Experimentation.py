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

def main(erdos_renyi_test=True, ring_k_test=True):
    '''
    Main loop for testing within this Python file
    '''
    # initialise variables
    N = 1000
    tMax = 30
    maxalpha = 0.4
    maxgamma = 0.1
    maxbeta = 4
    t = np.linspace(0,tMax,1000)

    # iterate through populations for complete graphs
    print("Beginning toy network simulations")

    if (erdos_renyi_test):
        print("Beginning Erdos-Renyi prob. tests")
        network = ig.Graph.Full(1000)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            tfulli, Sfulli, Ifulli, Rfulli = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, tfulli, Sfulli, right=Sfulli[-1])
            I[i] = np.interp(t, tfulli, Ifulli, right=Ifulli[-1])
            R[i] = np.interp(t, tfulli, Rfulli, right=Rfulli[-1])
            plt.plot(tfulli, Sfulli, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(tfulli, Ifulli, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(tfulli, Rfulli, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        Sfull = np.median(np.array(list(S.values())),0)
        Ifull = np.median(np.array(list(I.values())),0)
        Rfull = np.median(np.array(list(R.values())),0)
        plt.plot(t, Sfull, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, Ifull, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, Rfull, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a fully connected population size of {N}", fontsize=20)
        plt.savefig(f"PythonPlotting/Erdos_Renyi_Tests/Fully_Connected")

        network = ig.Graph.Erdos_Renyi(1000,0.5)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t5i, S5i, I5i, R5i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t5i, S5i, right=S5i[-1])
            I[i] = np.interp(t, t5i, I5i, right=I5i[-1])
            R[i] = np.interp(t, t5i, R5i, right=R5i[-1])
            plt.plot(t5i, S5i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t5i, I5i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t5i, R5i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S5 = np.median(np.array(list(S.values())),0)
        I5 = np.median(np.array(list(I.values())),0)
        R5 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S5, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I5, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R5, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a connectivity of 0.5, population size of {N}", fontsize=20)
        plt.savefig(f"PythonPlotting/Erdos_Renyi_Tests/05_Connected")

        network = ig.Graph.Erdos_Renyi(1000,0.1)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t1i, S1i, I1i, R1i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t1i, S1i, right=S1i[-1])
            I[i] = np.interp(t, t1i, I1i, right=I1i[-1])
            R[i] = np.interp(t, t1i, R1i, right=R1i[-1])
            plt.plot(t1i, S1i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t1i, I1i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t1i, R1i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S1 = np.median(np.array(list(S.values())),0)
        I1 = np.median(np.array(list(I.values())),0)
        R1 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S5, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I5, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R5, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a connectivity of 0.1, population size of {N}", fontsize=20)
        plt.savefig(f"PythonPlotting/Erdos_Renyi_Tests/01_Connected")

        network = ig.Graph.Erdos_Renyi(1000,0.01)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
      
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t01i, S01i, I01i, R01i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t01i, S01i, right=S01i[-1])
            I[i] = np.interp(t, t01i, I01i, right=I01i[-1])
            R[i] = np.interp(t, t01i, R01i, right=R01i[-1])
            plt.plot(t01i, S01i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t01i, I01i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t01i, R01i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S01 = np.median(np.array(list(S.values())),0)
        I01 = np.median(np.array(list(I.values())),0)
        R01 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S01, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I01, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R01, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a connectivity of 0.1, population size of {N}", fontsize=20)
        plt.savefig(f"PythonPlotting/Erdos_Renyi_Tests/001_Connected")

        fig = plt.figure()
        plt.plot(t, Ifull, color="red",label="Full",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I5, color="blue",label="P = 0.5",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I1, color="green",label="P = 0.1",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I01, color="yellow",label="P = 0.01",lw = 2,alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Infected Individuals", fontsize=16)
        plt.title(f"SIR model with varying probabilities of arcs existing", fontsize=20)
        plt.savefig(f"PythonPlotting/Erdos_Renyi_Tests/P_Comparison")

    if(ring_k_test):
        print("Beginning Lattice Ring k tests")
        # initialise variables
        N = 1000
        tMax = 30
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,tMax,1000)

        network = ig.Graph.Watts_Strogatz(1,1000,500,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            tfulli, Sfulli, Ifulli, Rfulli = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, tfulli, Sfulli, right=Sfulli[-1])
            I[i] = np.interp(t, tfulli, Ifulli, right=Ifulli[-1])
            R[i] = np.interp(t, tfulli, Rfulli, right=Rfulli[-1])
            plt.plot(tfulli, Sfulli, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(tfulli, Ifulli, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(tfulli, Rfulli, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        Sfull = np.median(np.array(list(S.values())),0)
        Ifull = np.median(np.array(list(I.values())),0)
        Rfull = np.median(np.array(list(R.values())),0)
        plt.plot(t, Sfull, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, Ifull, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, Rfull, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a fully connected ring lattice population {N}", fontsize=20)
        plt.savefig(f"PythonPlotting/Erdos_Renyi_Tests/Fully_Connected")

        network = ig.Graph.Watts_Strogatz(1,1000,400,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t4i, S4i, I4i, R4i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t4i, S4i, right=S4i[-1])
            I[i] = np.interp(t, t4i, I4i, right=I4i[-1])
            R[i] = np.interp(t, t4i, R4i, right=R4i[-1])
            plt.plot(t4i, S4i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t4i, I4i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t4i, R4i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S4 = np.median(np.array(list(S.values())),0)
        I4 = np.median(np.array(list(I.values())),0)
        R4 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S4, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I4, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R4, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a ring lattice population {N}, \n each node connected to nearest 400 neighbours", fontsize=20)
        plt.savefig(f"PythonPlotting/Lattice_K_Tests/400_Neighbours")

        network = ig.Graph.Watts_Strogatz(1,1000,200,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t2i, S2i, I2i, R2i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t2i, S2i, right=S2i[-1])
            I[i] = np.interp(t, t2i, I2i, right=I2i[-1])
            R[i] = np.interp(t, t2i, R2i, right=R2i[-1])
            plt.plot(t2i, S2i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t2i, I2i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t2i, R2i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S2 = np.median(np.array(list(S.values())),0)
        I2 = np.median(np.array(list(I.values())),0)
        R2 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S2, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I2, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R2, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a ring lattice population {N}, \n each node connected to nearest 200 neighbours", fontsize=20)
        plt.savefig(f"PythonPlotting/Lattice_K_Tests/200_Neighbours")

        network = ig.Graph.Watts_Strogatz(1,1000,100,0)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network,0.001)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t1i, S1i, I1i, R1i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t1i, S1i, right=S1i[-1])
            I[i] = np.interp(t, t1i, I1i, right=I1i[-1])
            R[i] = np.interp(t, t1i, R1i, right=R1i[-1])
            plt.plot(t1i, S1i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t1i, I1i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t1i, R1i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S1 = np.median(np.array(list(S.values())),0)
        I1 = np.median(np.array(list(I.values())),0)
        R1 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S1, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I1, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R1, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a ring lattice population {N}, \n each node connected to nearest 100 neighbours", fontsize=20)
        plt.savefig(f"PythonPlotting/Lattice_K_Tests/100_Neighbours")

        fig = plt.figure()
        plt.plot(t, Ifull, color="red",label="Full",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4, color="blue",label="Neighbourhood = 400",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I2, color="green",label="Neighbourhood = 200",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1, color="yellow",label="Neighbourhood = 100",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Infected Individuals", fontsize=16)
        plt.title(f"SIR model of ring lattice with varying neighbourhood sizes", fontsize=20)
        plt.savefig(f"PythonPlotting/Lattice_K_Tests/K_Comparison")

if __name__=="__main__":
    main(False, False)
