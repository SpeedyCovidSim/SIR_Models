import numpy as np
from numpy import random
import time
import igraph as ig
import copy
from matplotlib import pyplot as plt
from pythonCompartment.sirNetworksFrequency import gillespieDirectNetwork


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
    print("Beginning toy network simulations")
    if (er_small_test):
        # initialise variables
        N = 1000
        tMax = 30
        maxalpha = 0.4
        maxbeta = 4
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

        network = ig.Graph.Erdos_Renyi(1000,0.005)
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
        plt.title(f"SIR model with an arc prob. of 0.005, population size of {N}", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_ER_Tests/005_Connected")

        network = ig.Graph.Erdos_Renyi(1000,0.003)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t3i, S3i, I3i, R3i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t3i, S3i, right=S3i[-1])
            I[i] = np.interp(t, t3i, I3i, right=I3i[-1])
            R[i] = np.interp(t, t3i, R3i, right=R3i[-1])
            plt.plot(t3i, S3i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t3i, I3i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t3i, R3i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S3 = np.median(np.array(list(S.values())),0)
        I3 = np.median(np.array(list(I.values())),0)
        R3 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S3, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I3, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R3, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with an arc prob. of 0.003, population size of {N}", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_ER_Tests/003_Connected")

        network = ig.Graph.Erdos_Renyi(1000,0.001)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
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
        plt.title(f"SIR model with an arc prob. of 0.001, population size of {N}", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_ER_Tests/001_Connected")

        fig = plt.figure()
        plt.plot(t, Ifull, color="red",label="Full",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I5, color="blue",label="P = 0.005",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I3, color="green",label="P = 0.003",lw = 2,alpha=0.5,figure=fig)
        plt.plot(t, I1, color="yellow",label="P = 0.001",lw = 2,alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Infected Individuals", fontsize=16)
        plt.title(f"SIR model with varying probabilities of arcs existing", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_ER_Tests/P_Comparison")

    if(ring_small_test):
        print("Beginning Small Ring k tests")
        # initialise variables
        N = 1000
        tMax = 30
        maxalpha = 0.4
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
        plt.savefig(f"PythonPlotting/Small_K_Tests/Fully_Connected")

        network = ig.Graph.Watts_Strogatz(1,1000,40,0)
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
        plt.title(f"SIR model with a ring lattice population {N}, \n each node connected to nearest 40 neighbours", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_K_Tests/40_Neighbours")

        network = ig.Graph.Watts_Strogatz(1,1000,20,0)
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
        plt.title(f"SIR model with a ring lattice population {N}, \n each node connected to nearest 20 neighbours", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_K_Tests/20_Neighbours")

        network = ig.Graph.Watts_Strogatz(1,1000,10,0)
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
        plt.title(f"SIR model with a ring lattice population {N}, \n each node connected to nearest 10 neighbours", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_K_Tests/10_Neighbours")

        network = ig.Graph.Watts_Strogatz(1,1000,5,0)
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
            t05i, S05i, I05i, R05i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t05i, S05i, right=S05i[-1])
            I[i] = np.interp(t, t05i, I05i, right=I05i[-1])
            R[i] = np.interp(t, t05i, R05i, right=R05i[-1])
            plt.plot(t05i, S05i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t05i, I05i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t05i, R05i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S05 = np.median(np.array(list(S.values())),0)
        I05 = np.median(np.array(list(I.values())),0)
        R05 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S05, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I05, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R05, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a ring lattice population {N}, \n each node connected to nearest 5 neighbours", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_K_Tests/5_Neighbours")

        network = ig.Graph.Watts_Strogatz(1,1000,2,0)
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
            t02i, S02i, I02i, R02i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t02i, S02i, right=S02i[-1])
            I[i] = np.interp(t, t02i, I02i, right=I02i[-1])
            R[i] = np.interp(t, t02i, R02i, right=R02i[-1])
            plt.plot(t02i, S02i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t02i, I02i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t02i, R02i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S02 = np.median(np.array(list(S.values())),0)
        I02 = np.median(np.array(list(I.values())),0)
        R02 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S02, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I02, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R02, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Individuals in State", fontsize=16)
        plt.title(f"SIR model with a ring lattice population {N}, \n each node connected to nearest 2 neighbours", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_K_Tests/2_Neighbours")

        fig = plt.figure()
        plt.plot(t, Ifull, color="red",label="Full",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4, color="blue",label="Neighbourhood = 40",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I2, color="green",label="Neighbourhood = 20",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1, color="yellow",label="Neighbourhood = 10",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I05, color="black",label="Neighbourhood = 5",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I02, color="orange",label="Neighbourhood = 2",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Number of Infected Individuals", fontsize=16)
        plt.title(f"SIR model of ring lattice with varying neighbourhood sizes", fontsize=20)
        plt.savefig(f"PythonPlotting/Small_K_Tests/K_Comparison")


if __name__=="__main__":
    main(False, False)
