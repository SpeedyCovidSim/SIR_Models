import numpy as np
from numpy import random
import time
import igraph as ig
import copy
from matplotlib import pyplot as plt
from pythonCompartment.sirNetworksFrequency import gillespieDirectNetwork

def create_linked_neighbourhood(N, n):
    '''
    Creates N full subgraphs of n people with 1 link between adjacent subgraphs
    '''
    network = ig.Graph.Full(n)
    for i in range(1,N):
        networki = ig.Graph.Full(n)
        network = ig.operators.disjoint_union([network,networki])
        if(i < N-1):
            network.add_edge(n*i-1,n*i)
        else:
            network.add_edge(0,n*(i+1)-1)
            network.add_edge(n*i-1,n*i)
    return network
        
def create_random_neighbourhood(N, n, k):
    '''
    Creates N full subgraphs of n people with k random links also in the network
    (e.g. testing for multiple random links ala AL3/AL4 spread)
    note k < (n-1)N else this results in a fully-connected network
    '''
    network = create_linked_neighbourhood(N, n)
    pops = np.arange(N*n)
    for i in range(k):
        inserted = False
        while not(inserted):
            N_i = random.choice(N)
            out_nodes = np.arange(N_i*n,(N_i+1)*n-1)
            in_nodes = np.setdiff1d(pops,out_nodes)
            out_node = random.choice(out_nodes)
            in_node = random.choice(in_nodes)
            if not(network.are_connected(out_node,in_node)):
                network.add_edge(out_node,in_node)
                inserted = True
    return network

def create_working_neighbourhood(N, n, k):
    '''
    Creates N full subgraphs of n people with 1 person connected to k others
    (e.g. increasing number of people in a workplace)
    note k < 2(N-1) (for workplace simulation, otherwise end up with more 
                     'random' spreading between otherwise unrelated people)
    '''
    network = create_linked_neighbourhood(N, n)
    for i in range(1,k+1):
        if i < N:
            network.add_edge(0,i*n)
        else:
            network.add_edge(0,(i-N+2)*n-1)
    return network

def setNetwork_neighbourhoods(network, N, n, num_seeds=1):
    '''
    Initialises a parsed neighbourhood of full subgraphs and infects at most one individual per subgraph
    '''
    # get random proportion of populace to be infected and set states
    infecteds = np.zeros(num_seeds,dtype=int)
    num = N*n
    for i in range(num_seeds):
        infecteds[i] = random.choice(np.arange(i*n,(i+1)*n))
    infecteds = list(infecteds)
    susceptible = np.ones(num)
    susceptible[infecteds] = 0

    # set SIR numbers
    iTotal = num_seeds
    sTotal = num - num_seeds
    rTotal = 0

    # adding in hazards/rates
    numInfNei, numSusNei = initHazards(network, infecteds, num)
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
    Num = network.vcount()
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

def main(single_lattices=True, multi_lattices=True,k_random=True,k_workplace=True):
    '''
    Main loop for testing within this Python file
    '''
    print("Beginning toy network simulations")
    if(single_lattices):
        print("Beginning Single Seed Increasing Households tests")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,10,1000)

        N = 8
        n = 4
        num = N*n
        network = create_linked_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n 1 link between adjacent households")
        plt.savefig(f"PythonPlotting/Single_Household/{N}_Households")

        N = 16
        num = N*n
        network = create_linked_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n 1 link between adjacent households")
        plt.savefig(f"PythonPlotting/Single_Household/{N}_Households")

        N = 32
        num = N*n
        network = create_linked_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t8i, S8i, I8i, R8i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t8i, S8i, right=S4i[-1])
            I[i] = np.interp(t, t8i, I8i, right=I4i[-1])
            R[i] = np.interp(t, t8i, R8i, right=R4i[-1])
            plt.plot(t8i, S8i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t8i, I8i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t8i, R8i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S8 = np.median(np.array(list(S.values())),0)
        I8 = np.median(np.array(list(I.values())),0)
        R8 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S8, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I8, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R8, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n 1 link between adjacent households")
        plt.savefig(f"PythonPlotting/Single_Household/{N}_Households")

        N = 64
        num = N*n
        network = create_linked_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t16i, S16i, I16i, R16i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t16i, S16i, right=S16i[-1])
            I[i] = np.interp(t, t16i, I16i, right=I16i[-1])
            R[i] = np.interp(t, t16i, R16i, right=R16i[-1])
            plt.plot(t16i, S16i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t16i, I16i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t16i, R16i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S16 = np.median(np.array(list(S.values())),0)
        I16 = np.median(np.array(list(I.values())),0)
        R16 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S16, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I16, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R16, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n 1 link between adjacent households")
        plt.savefig(f"PythonPlotting/Single_Household/{N}_Households")

        fig = plt.figure()
        plt.plot(t, I2, color="blue",label="Neighbourhoods = 8",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4, color="green",label="Neighbourhoods = 16",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8, color="red",label="Neighbourhoods = 32",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I16, color="black",label="Neighbourhoods = 64",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of ring lattice with varying neighbourhood sizes")
        plt.savefig(f"PythonPlotting/Single_Household/N_Comparison")

    if(multi_lattices):
        print("Beginning Multi Seed Increasing Households tests")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,tMax,1000)

        N = 8
        n = 4
        num = N*n
        network = create_linked_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,int(N/2))
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n 1 link between adjacent households")
        plt.savefig(f"PythonPlotting/Multi_Household/{N}_Households")

        N = 16
        num = N*n
        network = create_linked_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,int(N/2))
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n 1 link between adjacent households")
        plt.savefig(f"PythonPlotting/Multi_Household/{N}_Households")

        N = 32
        num = N*n
        network = create_linked_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,int(N/2))
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t8i, S8i, I8i, R8i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t8i, S8i, right=S8i[-1])
            I[i] = np.interp(t, t8i, I8i, right=I8i[-1])
            R[i] = np.interp(t, t8i, R8i, right=R8i[-1])
            plt.plot(t8i, S8i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t8i, I8i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t8i, R8i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S8 = np.median(np.array(list(S.values())),0)
        I8 = np.median(np.array(list(I.values())),0)
        R8 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S8, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I8, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R8, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n 1 link between adjacent households")
        plt.savefig(f"PythonPlotting/Multi_Household/{N}_Households")

        N = 64
        num = N*n
        network = create_linked_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,int(N/2))
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            t16i, S16i, I16i, R16i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t16i, S16i, right=S16i[-1])
            I[i] = np.interp(t, t16i, I16i, right=I16i[-1])
            R[i] = np.interp(t, t16i, R16i, right=R16i[-1])
            plt.plot(t16i, S16i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t16i, I16i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t16i, R16i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S16 = np.median(np.array(list(S.values())),0)
        I16 = np.median(np.array(list(I.values())),0)
        R16 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S16, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I16, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R16, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n 1 link between adjacent households")
        plt.savefig(f"PythonPlotting/Multi_Household/{N}_Households")

        fig = plt.figure()
        plt.plot(t, I2, color="blue",label="Neighbourhoods = 8",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4, color="green",label="Neighbourhoods = 16",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8, color="red",label="Neighbourhoods = 32",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I16, color="black",label="Neighbourhoods = 64",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of ring lattice with varying neighbourhood sizes")
        plt.savefig(f"PythonPlotting/Multi_Household/N_Comparison")

    if(k_random):
        print("Beginning k Random Inter-Household Links Testing")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,tMax,1000)

        N = 7
        n = 4
        num = N*n
        k = 1
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            network = create_random_neighbourhood(N,n,k)
            iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
            rates = np.zeros(2*num)
            for inf in infecteds:
                # set recovery hazard
                rates[num+inf] = maxalpha
                # set infection hazard
                rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n {k} random links between households")
        plt.savefig(f"PythonPlotting/K_Random/{k}_Random")
        plt.close()


        k = 2
        S = {}
        I = {}
        R = {}
        fig = plt.figure()
        for i in range(30):
            network = create_random_neighbourhood(N,n,k)
            iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
            rates = np.zeros(2*num)
            for inf in infecteds:
                # set recovery hazard
                rates[num+inf] = maxalpha
                # set infection hazard
                rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n {k} random links between households")
        plt.savefig(f"PythonPlotting/K_Random/{k}_Random")
        plt.close()

        k = 5
        fig = plt.figure()
        S = {}
        I = {}
        R = {}
        for i in range(30):
            network = create_random_neighbourhood(N,n,k)
            iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
            rates = np.zeros(2*num)
            for inf in infecteds:
                # set recovery hazard
                rates[num+inf] = maxalpha
                # set infection hazard
                rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n {k} random links between households")
        plt.savefig(f"PythonPlotting/K_Random/{k}_Random")
        plt.close()

        k = 7
        fig = plt.figure()
        S = {}
        I = {}
        R = {}
        for i in range(30):
            network = create_random_neighbourhood(N,n,k)
            iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
            rates = np.zeros(2*num)
            for inf in infecteds:
                # set recovery hazard
                rates[num+inf] = maxalpha
                # set infection hazard
                rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
            t7i, S7i, I7i, R7i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t7i, S7i, right=S7i[-1])
            I[i] = np.interp(t, t7i, I7i, right=I7i[-1])
            R[i] = np.interp(t, t7i, R7i, right=R7i[-1])
            plt.plot(t7i, S7i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t7i, I7i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t7i, R7i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S7 = np.median(np.array(list(S.values())),0)
        I7 = np.median(np.array(list(I.values())),0)
        R7 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S1, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I1, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R1, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n {k} random links between households")
        plt.savefig(f"PythonPlotting/K_Random/{k}_Random")
        plt.close()

        fig = plt.figure()
        plt.plot(t, I1, color="blue",label="k = 1",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I2, color="green",label="k = 2",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5, color="red",label="k = 5",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I7, color="black",label="k = 7",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of neighbourhoods with varying random links")
        plt.savefig(f"PythonPlotting/K_Random/K_Comparison")
        plt.close()

    if(k_workplace):
        print("Beginning k Workplace Links Testing")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,tMax,1000)

        N = 7
        n = 4
        num = N*n
        k = 1
        network = create_working_neighbourhood(N,n,k)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        fig = plt.figure()
        S = {}
        I = {}
        R = {}
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n {k} workplace links between households")
        plt.savefig(f"PythonPlotting/K_Workplace/{k}_Workplace")
        plt.close()


        k = 2
        fig = plt.figure()
        network = create_working_neighbourhood(N,n,k)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n {k} workplace links between households")
        plt.savefig(f"PythonPlotting/K_Workplace/{k}_Workplace")
        plt.close()

        k = 5
        network = create_working_neighbourhood(N,n,k)
        fig = plt.figure()
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        S = {}
        I = {}
        R = {}
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
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n {k} workplace links between households")
        plt.savefig(f"PythonPlotting/K_Workplace/{k}_Workplace")
        plt.close()

        k = 7
        network = create_working_neighbourhood(N,n,k)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        fig = plt.figure()
        S = {}
        I = {}
        R = {}
        for i in range(30):
            t7i, S7i, I7i, R7i = gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            S[i] = np.interp(t, t7i, S7i, right=S7i[-1])
            I[i] = np.interp(t, t7i, I7i, right=I7i[-1])
            R[i] = np.interp(t, t7i, R7i, right=R7i[-1])
            plt.plot(t7i, S7i, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t7i, I7i, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
            plt.plot(t7i, R7i, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
        S7 = np.median(np.array(list(S.values())),0)
        I7 = np.median(np.array(list(I.values())),0)
        R7 = np.median(np.array(list(R.values())),0)
        plt.plot(t, S1, color="green",label="Susceptible",lw = 2,figure=fig)
        plt.plot(t, I1, color="red",label="Infected",lw = 2,figure=fig)
        plt.plot(t, R1, color="blue",label="Recovered",lw = 2,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Individuals in State")
        plt.title(f"SIR model with {N} households of {n} people, \n {k} workplace links between households")
        plt.savefig(f"PythonPlotting/K_Workplace/{k}_Workplace")
        plt.close()

        fig = plt.figure()
        plt.plot(t, I1, color="blue",label="k = 1",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I2, color="green",label="k = 2",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5, color="red",label="k = 5",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I7, color="black",label="k = 7",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of neighbourhoods with varying workplace links")
        plt.savefig(f"PythonPlotting/K_Workplace/K_Comparison")
        plt.close()


if __name__=="__main__":
    main(True, True, True, True)
