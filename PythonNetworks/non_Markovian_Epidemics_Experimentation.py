from Python_Nonhomog.Epidemic.SIRmax import gillespieMax
import numpy as np
from numpy import random
import igraph as ig
from matplotlib import pyplot as plt
from pythonCompartment.sirNetworksFrequency import gillespieDirectNetwork as gillespieSIR
from pythonCompartment.seirNetworksFrequency import gillespieSEIR
from PythonSimulation.simulate import nM_SIR_simulation, nM_SEIR_simulation

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

def main(non_SIR=True, non_SEIR=True,non_SIR_reco=True):
    '''
    Main loop for testing within this Python file
    '''
    # ignore true divide errors - there will be some cases division by 0 exists but this is handled
    np.seterr(divide='ignore', invalid='ignore')
    if (non_SIR):
        print("Beginning erdos-renyi nMarkov simulations")
        # initialise variables
        N = 1000
        tMax = 20
        maxalpha = 0.4
        maxbeta = 4
        haz_params = [1.1, 0.2591, 1.1, 2.5909]
        haz_max = [10, 1]
        haz_params_strong = [1.4, 0.2743, 1.1, 2.5909]
        haz_max_strong = [60, 1]
        j = 10
        def h_func(eventType, numSusNei, num_neighbours, entryTime, kinf, laminf, krec, lamrec, simTime):
            if eventType=="I":
                rate = kinf/laminf*(((simTime-entryTime)/laminf)**(kinf-1)) * (numSusNei/num_neighbours) if num_neighbours > 0 else 0
                return rate
            else:
                return krec/lamrec*(((simTime-entryTime)/lamrec)**(krec-1))
    
        t = np.linspace(0,tMax,1000)
        # initialise variables
        network = ig.Graph.Full(N)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = "Markov. vs Non-Markov. for Fully Connected"
        fname = "PythonPlotting/nM_ER_Tests/Full"
        fname2 = "PythonPlotting/nM_ER_Tests/Full_Strong"
        Sfullm, Ifullm, Rfullm, Sfullnm, Ifullnm, Rfullnm, Sfullnms, Ifullnms, Rfullnms = nM_SIR_simulation(j, gillespieSIR, gillespieMax, h_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)
    
        print("beginning 5")
        network = ig.Graph.Erdos_Renyi(N,0.005)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.005, population size 1000"
        fname = "PythonPlotting/nM_ER_Tests/05"
        fname2 = "PythonPlotting/nM_ER_Tests/05_Strong"
        S5m, I5m, R5m, S5nm, I5nm, R5nm,S5nms, I5nms, R5nms = nM_SIR_simulation(j, gillespieSIR, gillespieMax, h_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)

        print("beginning 3")
        network = ig.Graph.Erdos_Renyi(N,0.003)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.003, population size 1000"
        fname = "PythonPlotting/nM_ER_Tests/03"
        fname2 = "PythonPlotting/nM_ER_Tests/03_Strong"
        S3m, I3m, R3m, S3nm, I3nm, R3nm, S3nms, I3nms, R3nms = nM_SIR_simulation(j, gillespieSIR, gillespieMax, h_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)

        print("beginning 1")
        network = ig.Graph.Erdos_Renyi(N,0.001)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.001, population size 1000"
        fname = "PythonPlotting/nM_ER_Tests/01"
        fname2 = "PythonPlotting/nM_ER_Tests/01_Strong"
        S1m, I1m, R1m, S1nm, I1nm, R1nm, S1nms, I1nms, R1nms = nM_SIR_simulation(j, gillespieSIR, gillespieMax, h_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)
        title = "Markov. vs Non-Markov. for arc prob. of 0.001, population size 1000"

        fig = plt.figure()
        plt.plot(t, Ifullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnm, color="red",linestyle="dashed",label="Full - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nm, color="blue",linestyle="dashed",label="P = 0.005  - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nm, color="green",linestyle="dashed",label="P = 0.003 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nm, color="orange",linestyle="dashed",label="P = 0.001 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_ER_Tests/Comp")
        plt.close()

        fig = plt.figure()
        plt.plot(t, Ifullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnms, color="red",linestyle="dashed",label="Full - Strong Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nms, color="blue",linestyle="dashed",label="P = 0.005  - Strong Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nms, color="green",linestyle="dashed",label="P = 0.003 - Strong Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nms, color="orange",linestyle="dashed",label="P = 0.001 - Strong Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_ER_Tests/Comp_strong")
        plt.close()

        fig = plt.figure()
        plt.plot(t, Ifullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnm, color="red",linestyle="dashed",label="Full - Weak Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnms, color="red",linestyle="dotted",label="Full - Strong Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nm, color="blue",linestyle="dashed",label="P = 0.005  - Weak Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nms, color="blue",linestyle="dotted",label="P = 0.005  - Strong Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nm, color="green",linestyle="dashed",label="P = 0.003 - Weak Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nms, color="green",linestyle="dotted",label="P = 0.003 - Strong Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nm, color="orange",linestyle="dashed",label="P = 0.001 - Weak Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nms, color="orange",linestyle="dotted",label="P = 0.001 - Strong Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_ER_Tests/Comp_Full")
        plt.savefig(f"PythonPlotting/Comparisons/Comp_Full")
        plt.close()


    if (non_SEIR):
        print("Beginning erdos-renyi SEIR nMarkov simulations")
        # initialise variables
        N = 1000
        tMax = 20
        maxalpha = 0.416667
        maxbeta = 6.66667
        maxgamma = 10
        haz_params = [1.1, 0.2591, 1.1, 2.5909]
        haz_max = [10, 1]
        haz_params_strong = [1.4, 0.2743, 1.1, 2.5909]
        haz_max_strong = [60, 1]
        j = 10
        def h_func(eventType, numSusNei, num_neighbours, entryTime, kinf, laminf, krec, lamrec, simTime):
            if eventType=="I":
                rate = kinf/laminf*(((simTime-entryTime)/laminf)**(kinf-1)) * (numSusNei/num_neighbours) if num_neighbours > 0 else 0
                return rate
            else:
                return krec/lamrec*(((simTime-entryTime)/lamrec)**(krec-1))
    
        t = np.linspace(0,tMax,1000)
        # initialise variables
        network = ig.Graph.Full(N)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(3*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = "Markov. vs Non-Markov. for Fully Connected"
        fname = "PythonPlotting/nM_SEIR_Tests/Full"
        fname2 = "PythonPlotting/nM_SEIR_Tests/Full_Strong"
        Sfullm, Efullm, Ifullm, Inffullm, Rfullm, Sfullnm, Ifullnm, Rfullnm, Sfullnms, Ifullnms, Rfullnms = nM_SEIR_simulation(j, gillespieSEIR, gillespieMax, h_func, tMax, network, 0, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, maxgamma, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)

        print("beginning 5")
        network = ig.Graph.Erdos_Renyi(N,0.005)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(3*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.005, population size 1000"
        fname = "PythonPlotting/nM_SEIR_Tests/05"
        fname2 = "PythonPlotting/nM_SEIR_Tests/05_strong"
        S5m, E5m, I5m, Inf5m, R5m, S5nm, I5nm, R5nm, S5nms, I5nms, R5nms = nM_SEIR_simulation(j, gillespieSEIR, gillespieMax, h_func, tMax, network, 0, iTotal, sTotal, rTotal, numSusNei, rates, susceptible,
        maxalpha, maxbeta, maxgamma, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)

        print("beginning 3")
        network = ig.Graph.Erdos_Renyi(N,0.003)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(3*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.003, population size 1000"
        fname = "PythonPlotting/nM_SEIR_Tests/03"
        fname2 = "PythonPlotting/nM_SEIR_Tests/03_strong"
        S3m, E3m, I3m, Inf3m, R3m, S3nm, I3nm, R3nm, S3nms, I3nms, R3nms = nM_SEIR_simulation(j, gillespieSEIR, gillespieMax, h_func, tMax, network, 0, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, maxgamma, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)

        print("beginning 1")
        network = ig.Graph.Erdos_Renyi(N,0.001)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(3*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.001, population size 1000"
        fname = "PythonPlotting/nM_SEIR_Tests/01"
        fname2 = "PythonPlotting/nM_SEIR_Tests/01_strong"
        S1m, E1m, I1m, Inf1m, R1m, S1nm, I1nm, R1nm, S1nms, I1nms, R1nms = nM_SEIR_simulation(j, gillespieSEIR, gillespieMax, h_func, tMax, network, 0, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, maxgamma, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)

        fig = plt.figure()
        plt.plot(t, Inffullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnm, color="red",linestyle="dashed",label="Full - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nm, color="blue",linestyle="dashed",label="P = 0.005  - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nm, color="green",linestyle="dashed",label="P = 0.003 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nm, color="orange",linestyle="dashed",label="P = 0.001 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_SEIR_Tests/Comp")
        plt.close()

        fig = plt.figure()
        plt.plot(t, Inffullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnms, color="red",linestyle="dashed",label="Full - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nms, color="blue",linestyle="dashed",label="P = 0.005  - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nms, color="green",linestyle="dashed",label="P = 0.003 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nms, color="orange",linestyle="dashed",label="P = 0.001 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_SEIR_Tests/Comp_strong")
        plt.close()


        fig = plt.figure()
        plt.plot(t, Inffullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnm, color="red",linestyle="dashed",label="Full - Weak Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnms, color="red",linestyle="dotted",label="Full - Strong Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nm, color="blue",linestyle="dashed",label="P = 0.005  - Weak Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nms, color="blue",linestyle="dotted",label="P = 0.005  - Strong Wiebull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nm, color="green",linestyle="dashed",label="P = 0.003 - Weak Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nms, color="green",linestyle="dotted",label="P = 0.003 - Strong Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Inf1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nm, color="orange",linestyle="dashed",label="P = 0.001 - Weak Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nms, color="orange",linestyle="dotted",label="P = 0.001 - Strong Weibull Infection",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_SEIR_Tests/Comp_Full")
        plt.savefig(f"PythonPlotting/Comparisons/SEIR_Comp_Full")
        plt.close()

    if (non_SIR_reco):
        print("Beginning erdos-renyi nMarkov simulations")
        # initialise variables
        N = 1000
        tMax = 20
        maxalpha = 0.4
        maxbeta = 4
        haz_params = [1.1, 0.2591, 1.1, 2.5909]
        haz_max = [10, 1]
        haz_params_strong = [1.1, 0.2591, 1.4, 1.6458]
        haz_max_strong = [10, 4]
        j = 10
        def h_func(eventType, numSusNei, num_neighbours, entryTime, kinf, laminf, krec, lamrec, simTime):
            if eventType=="I":
                rate = kinf/laminf*(((simTime-entryTime)/laminf)**(kinf-1)) * (numSusNei/num_neighbours) if num_neighbours > 0 else 0
                return rate
            else:
                return krec/lamrec*(((simTime-entryTime)/lamrec)**(krec-1))
    
        t = np.linspace(0,tMax,1000)
        # initialise variables
        network = ig.Graph.Full(N)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        title = "Markov. vs Non-Markov. for Fully Connected"
        fname = "PythonPlotting/nM_ER_R_Tests/Full"
        fname2 = "PythonPlotting/nM_ER_R_Tests/Full_Strong"
        Sfullm, Ifullm, Rfullm, Sfullnm, Ifullnm, Rfullnm, Sfullnms, Ifullnms, Rfullnms = nM_SIR_simulation(j, gillespieSIR, gillespieMax, h_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)
    
        print("beginning 5")
        network = ig.Graph.Erdos_Renyi(N,0.005)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.005, population size 1000"
        fname = "PythonPlotting/nM_ER_R_Tests/05"
        fname2 = "PythonPlotting/nM_ER_R_Tests/05_Strong"
        S5m, I5m, R5m, S5nm, I5nm, R5nm,S5nms, I5nms, R5nms = nM_SIR_simulation(j, gillespieSIR, gillespieMax, h_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)

        print("beginning 3")
        network = ig.Graph.Erdos_Renyi(N,0.003)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.003, population size 1000"
        fname = "PythonPlotting/nM_ER_R_Tests/03"
        fname2 = "PythonPlotting/nM_ER_R_Tests/03_Strong"
        S3m, I3m, R3m, S3nm, I3nm, R3nm, S3nms, I3nms, R3nms = nM_SIR_simulation(j, gillespieSIR, gillespieMax, h_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)

        print("beginning 1")
        network = ig.Graph.Erdos_Renyi(N,0.001)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        title = "Markov. vs Non-Markov. for arc prob. of 0.001, population size 1000"
        fname = "PythonPlotting/nM_ER_R_Tests/01"
        fname2 = "PythonPlotting/nM_ER_R_Tests/01_Strong"
        S1m, I1m, R1m, S1nm, I1nm, R1nm, S1nms, I1nms, R1nms = nM_SIR_simulation(j, gillespieSIR, gillespieMax, h_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, 
        maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2)
        title = "Markov. vs Non-Markov. for arc prob. of 0.001, population size 1000"

        fig = plt.figure()
        plt.plot(t, Ifullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnm, color="red",linestyle="dashed",label="Full - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nm, color="blue",linestyle="dashed",label="P = 0.005  - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nm, color="green",linestyle="dashed",label="P = 0.003 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nm, color="orange",linestyle="dashed",label="P = 0.001 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_ER_R_Tests/Comp")
        plt.close()

        fig = plt.figure()
        plt.plot(t, Ifullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnms, color="red",linestyle="dashed",label="Full - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nms, color="blue",linestyle="dashed",label="P = 0.005  - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nms, color="green",linestyle="dashed",label="P = 0.003 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nms, color="orange",linestyle="dashed",label="P = 0.001 - Weibull",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_ER_R_Tests/Comp_strong")
        plt.close()

        fig = plt.figure()
        plt.plot(t, Ifullm, color="red",label="Full - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnm, color="red",linestyle="dashed",label="Full - Weak Weibull Recovery",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullnms, color="red",linestyle="dotted",label="Full - Strong Weibull Recovery",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5m, color="blue",label="P = 0.005 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nm, color="blue",linestyle="dashed",label="P = 0.005  - Weak Weibull Recovery",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5nms, color="blue",linestyle="dotted",label="P = 0.005  - Strong Weibull Recovery",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3m, color="green",label="P = 0.003 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nm, color="green",linestyle="dashed",label="P = 0.003 - Weak Weibull Recovery",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3nms, color="green",linestyle="dotted",label="P = 0.003 - Strong Weibull Recovery",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1m, color="orange",label="P = 0.001 - Markov.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nm, color="orange",linestyle="dashed",label="P = 0.001 - Weak Weibull Recovery",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1nms, color="orange",linestyle="dotted",label="P = 0.001 - Strong Weibull Recovery",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Markov. vs Non-Markov. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/nM_ER_R_Tests/Comp_Full")
        plt.savefig(f"PythonPlotting/Comparisons/Comp_R_Full")
        plt.close()

        

if __name__=="__main__":
    main(True, True, True)
