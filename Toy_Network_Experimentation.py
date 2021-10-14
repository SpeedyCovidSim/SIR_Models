from os import replace
import numpy as np
from numpy import random
import igraph as ig
import copy
from matplotlib import pyplot as plt
from pythonCompartment.sirNetworksFrequency import gillespieDirectNetwork
from PythonSimulation.simulate import general_SIR_simulation, random_SIR_simulation, general_proportional_SIR_simulation

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
    Creates N full subgraphs of n people with each household connected
    to k others
    '''
    network = create_linked_neighbourhood(N, n)
    for i in range(1,k+1):
        if i < N:
            network.add_edge(0,i*n)
        else:
            network.add_edge(0,(i-N+2)*n-1)
    return network

def create_random_household_links(N, n, k):
    '''
    Creates N full subgraphs of n people with k random links 
    between households
    '''
    network = ig.Graph.Full(n)
    for i in range(1,N):
        networki = ig.Graph.Full(n)
        network = ig.operators.disjoint_union([network,networki])
    

    selections = np.arange(20)
    choices = []
    i = 0
    while i < k:
        houses = random.choice(selections,size=2,replace=False)
        if not(list(houses) in choices):
            network.add_edge(houses[0]*n,houses[1]*n+n-1)
            choices.append(list(houses))
            i += 1
    
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

def main(single_lattices=True, multi_lattices=True,k_random=True,k_workplace=True,strong_single_lattices=True,k_erdos_neighbourhood=True,strong_single_prop=True,multi_lattices_prop=True):
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
        j=30

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Single_Household/{N}_Households"
        S2, I2, R2 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Single_Household/{N}_Households"
        S4, I4, R4 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Single_Household/{N}_Households"
        S8, I8, R8 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Single_Household/{N}_Households"
        S16, I16, R16 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        fig = plt.figure()
        plt.plot(t, I2, color="blue",label="Number of Households = 8",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4, color="green",label="Number of Households = 16",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8, color="red",label="Number of Households = 32",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I16, color="black",label="Number of Households = 64",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of a varying number of single seeded households")
        plt.savefig(f"PythonPlotting/Single_Household/N_Comparison")
        plt.savefig(f"PythonPlotting/Comparisons/Single_N_Comparison")

    if(multi_lattices):
        print("Beginning Multi Seed Increasing Households tests")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,tMax,1000)
        j=30

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
        title=f"SIR model with {N} households of {n} people, \n Multi-seeding households"
        fname=f"PythonPlotting/Multi_Household/{N}_Households"
        S8, I8, R8 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n Multi-seeding households"
        fname=f"PythonPlotting/Multi_Household/{N}_Households"
        S16, I16, R16 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n Multi-seeding households"
        fname=f"PythonPlotting/Multi_Household/{N}_Households"
        S32, I32, R32 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n Multi-seeding households"
        fname=f"PythonPlotting/Multi_Household/{N}_Households"
        S64, I64, R64 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        fig = plt.figure()
        plt.plot(t, I8, color="blue",label="Number of Households = 8",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I16, color="green",label="Number of Households = 16",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I32, color="red",label="Number of Households = 32",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I64, color="black",label="Number of Households = 64",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of a varying number of multi-seeded households")
        plt.savefig(f"PythonPlotting/Multi_Household/N_Comparison")
        plt.savefig(f"PythonPlotting/Comparisons/Multi_N_Comparison")

    if(k_random):
        print("Beginning k Random Inter-Household Links Testing")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,tMax,1000)
        j=30

        N = 7
        n = 4
        num = N*n
        k = 1
        title=f"SIR model with {N} households of {n} people, \n {k} random links between households"
        fname=f"PythonPlotting/K_Random/{k}_Random"
        S1, I1, R1 = random_SIR_simulation(j,N,n,k, create_random_neighbourhood, setNetwork_neighbourhoods, 
        gillespieDirectNetwork, tMax, maxalpha, maxbeta, title, fname)

        k = 2
        title=f"SIR model with {N} households of {n} people, \n {k} random links between households"
        fname=f"PythonPlotting/K_Random/{k}_Random"
        S2, I2, R2 = random_SIR_simulation(j,N,n,k, create_random_neighbourhood, setNetwork_neighbourhoods, 
        gillespieDirectNetwork, tMax, maxalpha, maxbeta, title, fname)

        k = 5
        title=f"SIR model with {N} households of {n} people, \n {k} random links between households"
        fname=f"PythonPlotting/K_Random/{k}_Random"
        S5, I5, R5 = random_SIR_simulation(j,N,n,k, create_random_neighbourhood, setNetwork_neighbourhoods, 
        gillespieDirectNetwork, tMax, maxalpha, maxbeta, title, fname)

        k = 7
        title=f"SIR model with {N} households of {n} people, \n {k} random links between households"
        fname=f"PythonPlotting/K_Random/{k}_Random"
        S7, I7, R7 = random_SIR_simulation(j,N,n,k, create_random_neighbourhood, setNetwork_neighbourhoods, 
        gillespieDirectNetwork, tMax, maxalpha, maxbeta, title, fname)

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
        plt.savefig(f"PythonPlotting/Comparisons/K_Random_Comparison")
        plt.close()

    if(k_workplace):
        print("Beginning k Workplace Links Testing")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,tMax,1000)
        j=30

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
        title=f"SIR model with {N} households of {n} people, \n {k} workplace links between households"
        fname=f"PythonPlotting/K_Workplace/{k}_Workplace"
        S1, I1, R1 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n {k} workplace links between households"
        fname=f"PythonPlotting/K_Workplace/{k}_Workplace"
        S2, I2, R2 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n {k} workplace links between households"
        fname=f"PythonPlotting/K_Workplace/{k}_Workplace"
        S5, I5, R5 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n {k} workplace links between households"
        fname=f"PythonPlotting/K_Workplace/{k}_Workplace"
        S7, I7, R7 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)
        
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
        plt.savefig(f"PythonPlotting/Comparisons/K_Workplace_Comparison")
        plt.close()

    if(strong_single_lattices):
        print("Beginning Single Seed Increasing Households tests")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 48
        t = np.linspace(0,10,1000)
        j=30

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Strong_Single_Household/{N}_Households"
        S2, I2, R2 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Strong_Single_Household/{N}_Households"
        S4, I4, R4 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Strong_Single_Household/{N}_Households"
        S8, I8, R8 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Strong_Single_Household/{N}_Households"
        S16, I16, R16 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        fig = plt.figure()
        plt.plot(t, I2, color="blue",label="Number of Households = 8",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4, color="green",label="Number of Households = 16",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8, color="red",label="Number of Households = 32",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I16, color="black",label="Number of Households = 64",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of a varying number of single seeded households")
        plt.savefig(f"PythonPlotting/Strong_Single_Household/N_Comparison")
        plt.savefig(f"PythonPlotting/Comparisons/Strong_Single_N_Comparison")

    if(k_erdos_neighbourhood):
        print("Beginning k erdos neighbourhood")
        # initialise variables
        tMax = 20
        maxalpha = 0.4
        maxbeta = 6
        t = np.linspace(0,tMax,1000)
        j=30

        N = 100
        n = 5
        seed = 10
        num = N*n
        k = 10
        network = create_random_household_links(N,n,k)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,seed)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        fig = plt.figure()
        title=f"SIR model with {N} households of {n} people, \n {k} random links in network"
        fname=f"PythonPlotting/K_Household/{k}_Household"
        S1, I1, R1 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        k = 20
        fig = plt.figure()
        network = create_random_household_links(N,n,k)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,seed)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title=f"SIR model with {N} households of {n} people, \n {k} random links in network"
        fname=f"PythonPlotting/K_Household/{k}_Household"
        S2, I2, R2 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        k = 30
        network = create_random_household_links(N,n,k)
        fig = plt.figure()
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,seed)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title=f"SIR model with {N} households of {n} people, \n {k} random links in network"
        fname=f"PythonPlotting/K_Household/{k}_Household"
        S5, I5, R5 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        k = 40
        network = create_random_household_links(N,n,k)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,seed)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        fig = plt.figure()
        title=f"SIR model with {N} households of {n} people, \n {k} random links in network"
        fname=f"PythonPlotting/K_Household/{k}_Household"
        S7, I7, R7 = general_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)
        
        fig = plt.figure()
        plt.plot(t, I1, color="blue",label="k = 10",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I2, color="green",label="k = 20",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5, color="red",label="k = 30",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I7, color="black",label="k = 40",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of neighbourhoods with varying random links in network")
        plt.savefig(f"PythonPlotting/K_Household/K_Comparison")
        plt.savefig(f"PythonPlotting/Comparisons/K_Household_Comparison")
        plt.close()

    if(strong_single_prop):
        print("Beginning Single Seed Increasing Households tests")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 48
        t = np.linspace(0,10,1000)
        j=30

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Strong_Single_Prop/{N}_Households"
        S2, I2, R2 = general_proportional_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Strong_Single_Prop/{N}_Households"
        S4, I4, R4 = general_proportional_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Strong_Single_Prop/{N}_Households"
        S8, I8, R8 = general_proportional_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n 1 link between adjacent households"
        fname=f"PythonPlotting/Strong_Single_Prop/{N}_Households"
        S16, I16, R16 = general_proportional_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        fig = plt.figure()
        plt.plot(t, I2, color="blue",label="Number of Households = 8",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4, color="green",label="Number of Households = 16",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8, color="red",label="Number of Households = 32",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I16, color="black",label="Number of Households = 64",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Proportion of Infected Individuals")
        plt.title(f"SIR model of a varying number of single seeded households")
        plt.savefig(f"PythonPlotting/Strong_Single_Prop/N_Comparison")
        plt.savefig(f"PythonPlotting/Comparisons/Strong_Single_Prop_Comparison")

    if(multi_lattices_prop):
        print("Beginning Multi Seed Increasing Households tests")
        # initialise variables
        tMax = 15
        maxalpha = 0.4
        maxgamma = 0.1
        maxbeta = 4
        t = np.linspace(0,tMax,1000)
        j=30

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
        title=f"SIR model with {N} households of {n} people, \n Multi-seeding households"
        fname=f"PythonPlotting/Multi_Household_Prop/{N}_Households"
        S8, I8, R8 = general_proportional_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n Multi-seeding households"
        fname=f"PythonPlotting/Multi_Household_Prop/{N}_Households"
        S16, I16, R16 = general_proportional_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n Multi-seeding households"
        fname=f"PythonPlotting/Multi_Household_Prop/{N}_Households"
        S32, I32, R32 = general_proportional_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

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
        title=f"SIR model with {N} households of {n} people, \n Multi-seeding households"
        fname=f"PythonPlotting/Multi_Household_Prop/{N}_Households"
        S64, I64, R64 = general_proportional_SIR_simulation(j, gillespieDirectNetwork, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        fig = plt.figure()
        plt.plot(t, I8, color="blue",label="Number of Households = 8",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I16, color="green",label="Number of Households = 16",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I32, color="red",label="Number of Households = 32",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I64, color="black",label="Number of Households = 64",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"SIR model of a varying number of multi-seeded households")
        plt.savefig(f"PythonPlotting/Multi_Household_Prop/N_Comparison")
        plt.savefig(f"PythonPlotting/Comparisons/Multi_N_Comparison_prop")

if __name__=="__main__":
    main(False, False, False, False,False,False,True,True)
