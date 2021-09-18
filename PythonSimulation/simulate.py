import numpy as np
from matplotlib import pyplot as plt
import copy

def freq_dens_simulation(k, freq_func, dens_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta, title, fname):
    '''
    Method to run and plot freq. vs. dens. experimentation
    Returns the median SIR values for freq. and dens. scalings 
    '''
    t = np.linspace(0,tMax,1000)
    Sfreq = {}
    Ifreq = {}
    Rfreq = {}
    Sdens = {}
    Idens = {}
    Rdens = {}
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    for i in range(10):
        ti, Si, Ii, Ri = freq_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        Sfreq[i] = np.interp(t, ti, Si, right=Si[-1])
        Ifreq[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rfreq[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax1.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax1.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax1.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        ti, Si, Ii, Ri = dens_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        Sdens[i] = np.interp(t, ti, Si, right=Si[-1])
        Idens[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rdens[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax2.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax2.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax2.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
    Sfreq = np.median(np.array(list(Sfreq.values())),0)
    Ifreq = np.median(np.array(list(Ifreq.values())),0)
    Rfreq = np.median(np.array(list(Rfreq.values())),0)
    Sdens = np.median(np.array(list(Sdens.values())),0)
    Idens = np.median(np.array(list(Idens.values())),0)
    Rdens = np.median(np.array(list(Rdens.values())),0)
    line_labels = ["Susceptible", "Infected", "Recovered"]
    S = ax1.plot(t, Sfreq, color="green",lw = 2)[0]
    I = ax1.plot(t, Ifreq, color="red",lw = 2)[0]
    R = ax1.plot(t, Rfreq, color="blue",lw = 2)[0]
    ax2.plot(t, Sdens, color="green",lw = 2)
    ax2.plot(t, Idens, color="red",lw = 2)
    ax2.plot(t, Rdens, color="blue",lw = 2)
    ax1.set_xlabel('Time')
    ax2.set_xlabel('Time')
    ax1.set_ylabel('Number of Individuals in State')
    fig.legend(handles=[S, I, R],labels=line_labels,loc="center right")
    fig.suptitle(title)
    plt.savefig(fname)
    return Sfreq,Ifreq,Rfreq,Sdens,Idens,Rdens