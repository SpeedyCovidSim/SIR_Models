import numpy as np
from numpy import random
import time
import igraph as ig
import copy
from matplotlib import pyplot as plt
from pythonCompartment.sirNetworksFrequency import gillespieDirectNetwork as gillespieFrequency
from pythonCompartment.sirNetworksDensity import gillespieDirectNetwork as gillespieDensity
from PythonSimulation.simulate import freq_dens_simulation

def create_fixed_neighbourhood(N, n):
    '''
    Creates N full subgraphs of n people
    '''
    network = ig.Graph.Full(n)
    for i in range(1,N):
        networki = ig.Graph.Full(n)
        network = ig.operators.disjoint_union([network,networki])
    return network

def create_random_neighbourhood(N, n):
    '''
    Creates N full subgraphs of a left-truncated Poiss(n) people with at least 2 people in a household (to get spread)
    '''
    house_sizes = random.poisson(n, N)
    house_sizes = [a if a>1 else 2 for a in house_sizes]
    network = ig.Graph.Full(house_sizes[0])

    for i in range(1,N):
        networki = ig.Graph.Full(house_sizes[i])
        network = ig.operators.disjoint_union([network,networki])
    return network, house_sizes
    

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

def setNetwork_random(network, N, n, house_sizes, num_seeds):
    '''
    Initialises a parsed neighbourhood of full subgraphs and infects at most one individual per subgraph
    '''
    # get random proportion of populace to be infected and set states
    infecteds = np.zeros(num_seeds,dtype=int)
    num = np.sum(house_sizes)
    for i in range(num_seeds):
        infecteds[i] = random.choice(house_sizes[i]) + np.sum(house_sizes[:i])
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

def main(er_small_test=True, household_test=True, random_test=True):
    '''
    Main loop for testing within this Python file
    '''
    if (er_small_test):
        print("Beginning household density vs frequency network simulations")
        # initialise variables
        N = 1000
        tMax = 20
        maxalpha = 0.4
        maxbeta = 4
        t = np.linspace(0,tMax,1000)
        j = 30

        # initialise variables
        network = ig.Graph.Full(N)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = "Freq. vs Dens. for Fully Connected"
        fname = "PythonPlotting/Freq_Dens_Tests/Full"
        Sfullfreq, Ifullfreq, Rfullfreq, Sfulldens, Ifulldens, Rfulldens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Erdos_Renyi(N,0.005)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Freq. vs Dens. for arc prob. of 0.005, population size 1000"
        fname = "PythonPlotting/Freq_Dens_Tests/05"
        S5freq, I5freq, R5freq, S5dens, I5dens, R5dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Erdos_Renyi(N,0.003)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Freq. vs Dens. for arc prob. of 0.003, population size 1000"
        fname = "PythonPlotting/Freq_Dens_Tests/03"
        S3freq, I3freq, R3freq, S3dens, I3dens, R3dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        network = ig.Graph.Erdos_Renyi(N,0.001)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Freq. vs Dens. for arc prob. of 0.001, population size 1000"
        fname = "PythonPlotting/Freq_Dens_Tests/01"
        S1freq, I1freq, R1freq, S1dens, I1dens, R1dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        fig = plt.figure()
        plt.plot(t, Ifullfreq, color="red",label="Full - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifulldens, color="red",linestyle="dashed",label="Full - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5freq, color="blue",label="P = 0.005 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5dens, color="blue",linestyle="dashed",label="P = 0.005  - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3freq, color="green",label="P = 0.003 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3dens, color="green",linestyle="dashed",label="P = 0.003 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1freq, color="orange",label="P = 0.001 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1dens, color="orange",label="P = 0.001 - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Freq. vs. Dens. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/Freq_Dens_Tests/Comp")
        plt.savefig(f"PythonPlotting/Comparisons/FD_ER_Comp")

    if(household_test):
        print("Beginning density vs frequency household size simulations")
        # initialise variables
        tMax = 20
        maxalpha = 0.4
        maxbeta = 4
        t = np.linspace(0,tMax,1000)
        j=30

        N = 8
        n = 4
        num = N*n
        network = create_fixed_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,N)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"Freq. vs Dens. for {N} households of {n} people"
        fname = f"PythonPlotting/Freq_Dens_Household_Test/{n}"
        S4freq, I4freq, R4freq, S4dens, I4dens, R4dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        n = 6
        num = N*n
        network = create_fixed_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,N)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"Freq. vs Dens. for {N} households of {n} people"
        fname = f"PythonPlotting/Freq_Dens_Household_Test/{n}"
        S6freq, I6freq, R6freq, S6dens, I6dens, R6dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)


        n = 8
        num = N*n
        network = create_fixed_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,N)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"Freq. vs Dens. for {N} households of {n} people"
        fname = f"PythonPlotting/Freq_Dens_Household_Test/{n}"
        S8freq, I8freq, R8freq, S8dens, I8dens, R8dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)


        n = 10
        num = N*n
        network = create_fixed_neighbourhood(N,n)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_neighbourhoods(network,N,n,N)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"Freq. vs Dens. for {N} households of {n} people"
        fname = f"PythonPlotting/Freq_Dens_Household_Test/{n}"
        S10freq, I10freq, R10freq, S10dens, I10dens, R10dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)


        fig = plt.figure()
        plt.plot(t, I4freq, color="red",label="4 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4dens, color="red",linestyle="dashed",label="4 - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I6freq, color="blue",label="6 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I6dens, color="blue",linestyle="dashed",label="6  - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8freq, color="green",label="8 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8dens, color="green",linestyle="dashed",label="8 - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I10freq, color="orange",label="10 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I10dens, color="orange",label="10 - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Freq. vs. Dens. with varying household sizes")
        plt.savefig(f"PythonPlotting/Freq_Dens_Household_Test/Comp")
        plt.savefig(f"PythonPlotting/Comparisons/FD_Fixed_Household_Comp")

    if(random_test):
        print("Beginning density vs frequency household Poisson simulations")
        # initialise variables
        tMax = 20
        maxalpha = 0.4
        maxbeta = 4
        t = np.linspace(0,tMax,1000)
        j = 30

        N = 15
        n = 4
        network, house_sizes = create_random_neighbourhood(N,n)
        num = np.sum(house_sizes)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_random(network,N,n,house_sizes,N)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"Freq. vs Dens. for {N} households of Poi({n}) people"
        fname = f"PythonPlotting/Freq_Dens_Poisson_Test/{n}"
        S4freq, I4freq, R4freq, S4dens, I4dens, R4dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)

        n = 6
        network, house_sizes = create_random_neighbourhood(N,n)
        num = np.sum(house_sizes)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_random(network,N,n,house_sizes,N)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"Freq. vs Dens. for {N} households of Poi({n}) people"
        fname = f"PythonPlotting/Freq_Dens_Poisson_Test/{n}"
        S6freq, I6freq, R6freq, S6dens, I6dens, R6dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)


        n = 8
        network, house_sizes = create_random_neighbourhood(N,n)
        num = np.sum(house_sizes)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_random(network,N,n,house_sizes,N)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"Freq. vs Dens. for {N} households of Poi({n}) people"
        fname = f"PythonPlotting/Freq_Dens_Poisson_Test/{n}"
        S8freq, I8freq, R8freq, S8dens, I8dens, R8dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)


        n = 10
        network, house_sizes = create_random_neighbourhood(N,n)
        num = np.sum(house_sizes)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork_random(network,N,n,house_sizes,N)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = f"Freq. vs Dens. for {N} households of Poi({n}) people"
        fname = f"PythonPlotting/Freq_Dens_Poisson_Test/{n}"
        S10freq, I10freq, R10freq, S10dens, I10dens, R10dens = freq_dens_simulation(j, gillespieFrequency, gillespieDensity, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, title, fname)


        fig = plt.figure()
        plt.plot(t, I4freq, color="red",label="Poi(4) - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I4dens, color="red",linestyle="dashed",label="Poi(4) - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I6freq, color="blue",label="Poi(6) - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I6dens, color="blue",linestyle="dashed",label="Poi(6)  - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8freq, color="green",label="Poi(8) - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I8dens, color="green",linestyle="dashed",label="Poi(8) - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I10freq, color="orange",label="Poi(10) - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I10dens, color="orange",linestyle="dashed",label="Poi(10) - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Freq. vs. Dens. with varying Poi sizes")
        plt.savefig(f"PythonPlotting/Freq_Dens_Poisson_Test/Comp")
        plt.savefig(f"PythonPlotting/Comparisons/FD_Poisson_Household_Comp")

if __name__=="__main__":
    main(False, True, False)
