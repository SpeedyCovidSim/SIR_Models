import numpy as np
from numpy import random
import time
import igraph as ig
import copy
from matplotlib import pyplot as plt
from Python_Nonhomog.Epidemic.SIRmax import gillespieMax
from pythonCompartment.sirNetworksDensity import gillespieDirectNetwork as gillespieDensity


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

def main(er_small_test=True):
    '''
    Main loop for testing within this Python file
    '''
    if (er_small_test):
        print("Beginning density vs frequency network simulations")
        # initialise variables
        N = 1000
        tMax = 20
        maxalpha = 0.4
        maxbeta = 4
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
        Sfreq = {}
        Ifreq = {}
        Rfreq = {}
        Sdens = {}
        Idens = {}
        Rdens = {}
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
        for i in range(10):
            ti, Si, Ii, Ri = gillespieFrequency(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            Sfreq[i] = np.interp(t, ti, Si, right=Si[-1])
            Ifreq[i] = np.interp(t, ti, Ii, right=Ii[-1])
            Rfreq[i] = np.interp(t, ti, Ri, right=Ri[-1])
            ax1.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
            ax1.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
            ax1.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
            ti, Si, Ii, Ri = gillespieDensity(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            Sdens[i] = np.interp(t, ti, Si, right=Si[-1])
            Idens[i] = np.interp(t, ti, Ii, right=Ii[-1])
            Rdens[i] = np.interp(t, ti, Ri, right=Ri[-1])
            ax2.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
            ax2.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
            ax2.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        Sfullfreq = np.median(np.array(list(Sfreq.values())),0)
        Ifullfreq = np.median(np.array(list(Ifreq.values())),0)
        Rfullfreq = np.median(np.array(list(Rfreq.values())),0)
        Sfulldens = np.median(np.array(list(Sdens.values())),0)
        Ifulldens = np.median(np.array(list(Idens.values())),0)
        Rfulldens = np.median(np.array(list(Rdens.values())),0)
        line_labels = ["Susceptible", "Infected", "Recovered"]
        S = ax1.plot(t, Sfullfreq, color="green",lw = 2)[0]
        I = ax1.plot(t, Ifullfreq, color="red",lw = 2)[0]
        R = ax1.plot(t, Rfullfreq, color="blue",lw = 2)[0]
        ax2.plot(t, Sfulldens, color="green",lw = 2)
        ax2.plot(t, Ifulldens, color="red",lw = 2)
        ax2.plot(t, Rfulldens, color="blue",lw = 2)
        ax1.set_xlabel('Time')
        ax2.set_xlabel('Time')
        ax1.set_ylabel('Number of Individuals in State')
        fig.legend([S, I, R],labels=line_labels,loc="center right")
        fig.suptitle("Freq. vs Dens. for Fully Connected")
        plt.savefig(f"PythonPlotting/Freq_Dens_Tests/Full")

        network = ig.Graph.Erdos_Renyi(N,0.005)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        Sfreq = {}
        Ifreq = {}
        Rfreq = {}
        Sdens = {}
        Idens = {}
        Rdens = {}
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
        for i in range(10):
            ti, Si, Ii, Ri = gillespieFrequency(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            Sfreq[i] = np.interp(t, ti, Si, right=Si[-1])
            Ifreq[i] = np.interp(t, ti, Ii, right=Ii[-1])
            Rfreq[i] = np.interp(t, ti, Ri, right=Ri[-1])
            ax1.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
            ax1.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
            ax1.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
            ti, Si, Ii, Ri = gillespieDensity(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            Sdens[i] = np.interp(t, ti, Si, right=Si[-1])
            Idens[i] = np.interp(t, ti, Ii, right=Ii[-1])
            Rdens[i] = np.interp(t, ti, Ri, right=Ri[-1])
            ax2.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
            ax2.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
            ax2.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        S5freq = np.median(np.array(list(Sfreq.values())),0)
        I5freq = np.median(np.array(list(Ifreq.values())),0)
        R5freq = np.median(np.array(list(Rfreq.values())),0)
        S5dens = np.median(np.array(list(Sdens.values())),0)
        I5dens = np.median(np.array(list(Idens.values())),0)
        R5dens = np.median(np.array(list(Rdens.values())),0)
        line_labels = ["Susceptible", "Infected", "Recovered"]
        S = ax1.plot(t, S5freq, color="green",lw = 2)[0]
        I = ax1.plot(t, I5freq, color="red",lw = 2)[0]
        R = ax1.plot(t, R5freq, color="blue",lw = 2)[0]
        ax2.plot(t, S5dens, color="green",lw = 2)
        ax2.plot(t, I5dens, color="red",lw = 2)
        ax2.plot(t, R5dens, color="blue",lw = 2)
        ax1.set_xlabel('Time')
        ax2.set_xlabel('Time')
        ax1.set_ylabel('Number of Individuals in State')
        fig.legend([S, I, R],labels=line_labels,loc="center right")
        fig.suptitle("Freq. vs Dens. for arc prob. of 0.005, population size 1000")
        plt.savefig(f"PythonPlotting/Freq_Dens_Tests/05")

        network = ig.Graph.Erdos_Renyi(N,0.003)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        Sfreq = {}
        Ifreq = {}
        Rfreq = {}
        Sdens = {}
        Idens = {}
        Rdens = {}
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
        for i in range(10):
            ti, Si, Ii, Ri = gillespieFrequency(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            Sfreq[i] = np.interp(t, ti, Si, right=Si[-1])
            Ifreq[i] = np.interp(t, ti, Ii, right=Ii[-1])
            Rfreq[i] = np.interp(t, ti, Ri, right=Ri[-1])
            ax1.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
            ax1.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
            ax1.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
            ti, Si, Ii, Ri = gillespieDensity(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            Sdens[i] = np.interp(t, ti, Si, right=Si[-1])
            Idens[i] = np.interp(t, ti, Ii, right=Ii[-1])
            Rdens[i] = np.interp(t, ti, Ri, right=Ri[-1])
            ax2.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
            ax2.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
            ax2.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        S3freq = np.median(np.array(list(Sfreq.values())),0)
        I3freq = np.median(np.array(list(Ifreq.values())),0)
        R3freq = np.median(np.array(list(Rfreq.values())),0)
        S3dens = np.median(np.array(list(Sdens.values())),0)
        I3dens = np.median(np.array(list(Idens.values())),0)
        R3dens = np.median(np.array(list(Rdens.values())),0)
        line_labels = ["Susceptible", "Infected", "Recovered"]
        S = ax1.plot(t, S3freq, color="green",lw = 2)[0]
        I = ax1.plot(t, I3freq, color="red",lw = 2)[0]
        R = ax1.plot(t, R3freq, color="blue",lw = 2)[0]
        ax2.plot(t, S3dens, color="green",lw = 2)
        ax2.plot(t, I3dens, color="red",lw = 2)
        ax2.plot(t, R3dens, color="blue",lw = 2)
        ax1.set_xlabel('Time')
        ax2.set_xlabel('Time')
        ax1.set_ylabel('Number of Individuals in State')
        fig.legend([S, I, R],labels=line_labels,loc="center right")
        fig.suptitle("Freq. vs Dens. for arc prob. of 0.003, population size 1000")
        plt.savefig(f"PythonPlotting/Freq_Dens_Tests/03")

        network = ig.Graph.Erdos_Renyi(N,0.001)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = setNetwork(network)
        rates = np.zeros(2*N)
        for inf in infecteds:
            # set recovery hazard
            rates[N+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf) if network.degree(inf)>0 else 0
        Sfreq = {}
        Ifreq = {}
        Rfreq = {}
        Sdens = {}
        Idens = {}
        Rdens = {}
        fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
        for i in range(10):
            ti, Si, Ii, Ri = gillespieFrequency(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            Sfreq[i] = np.interp(t, ti, Si, right=Si[-1])
            Ifreq[i] = np.interp(t, ti, Ii, right=Ii[-1])
            Rfreq[i] = np.interp(t, ti, Ri, right=Ri[-1])
            ax1.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
            ax1.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
            ax1.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
            ti, Si, Ii, Ri = gillespieDensity(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
            Sdens[i] = np.interp(t, ti, Si, right=Si[-1])
            Idens[i] = np.interp(t, ti, Ii, right=Ii[-1])
            Rdens[i] = np.interp(t, ti, Ri, right=Ri[-1])
            ax2.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
            ax2.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
            ax2.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        S1freq = np.median(np.array(list(Sfreq.values())),0)
        I1freq = np.median(np.array(list(Ifreq.values())),0)
        R1freq = np.median(np.array(list(Rfreq.values())),0)
        S1dens = np.median(np.array(list(Sdens.values())),0)
        I1dens = np.median(np.array(list(Idens.values())),0)
        R1dens = np.median(np.array(list(Rdens.values())),0)
        line_labels = ["Susceptible", "Infected", "Recovered"]
        S = ax1.plot(t, S1freq, color="green",lw = 2)[0]
        I = ax1.plot(t, I1freq, color="red",lw = 2)[0]
        R = ax1.plot(t, R1freq, color="blue",lw = 2)[0]
        ax2.plot(t, S1dens, color="green",lw = 2)
        ax2.plot(t, I1dens, color="red",lw = 2)
        ax2.plot(t, R1dens, color="blue",lw = 2)
        ax1.set_xlabel('Time')
        ax2.set_xlabel('Time')
        ax1.set_ylabel('Number of Individuals in State')
        fig.legend([S, I, R],labels=line_labels,loc="center right")
        fig.suptitle("Freq. vs Dens. for arc prob. of 0.001, population size 1000")
        plt.savefig(f"PythonPlotting/Freq_Dens_Tests/01")

        fig = plt.figure()
        plt.plot(t, Ifulldens, color="red",label="Full - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, Ifullfreq, color="red",linestyle="dashed",label="Full - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5freq, color="blue",label="P = 0.005 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I5dens, color="blue",linestyle="dashed",label="P = 0.005  - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3freq, color="green",label="P = 0.003 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I3dens, color="green",linestyle="dashed",label="P = 0.003 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1freq, color="yellow",label="P = 0.001 - Freq.",lw = 2, alpha=0.5,figure=fig)
        plt.plot(t, I1dens, color="yellow",label="P = 0.001 - Dens.",lw = 2, alpha=0.5,figure=fig)
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Number of Infected Individuals")
        plt.title(f"Freq. vs. Dens. with varying probabilities of arcs existing")
        plt.savefig(f"PythonPlotting/Freq_Dens_Tests/Comp")

if __name__=="__main__":
    main(True)
