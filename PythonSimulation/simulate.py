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
    for i in range(k):
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

def general_SIR_simulation(k, direct_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta, title, fname):
    '''
    Method to run and plot general SIR experimentation
    Returns the median SIR values for parsed direct method
    '''
    S = {}
    I = {}
    R = {}
    fig = plt.figure()
    t = np.linspace(0,tMax,1000)
    for i in range(k):
        ti, Si, Ii, Ri = direct_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        S[i] = np.interp(t, ti, Si, right=Si[-1])
        I[i] = np.interp(t, ti, Ii, right=Ii[-1])
        R[i] = np.interp(t, ti, Ri, right=Ri[-1])
        plt.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
        plt.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
        plt.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
    S = np.median(np.array(list(S.values())),0)
    I = np.median(np.array(list(I.values())),0)
    R = np.median(np.array(list(R.values())),0)
    plt.plot(t, S, color="green",label="Susceptible",lw = 2,figure=fig)
    plt.plot(t, I, color="red",label="Infected",lw = 2,figure=fig)
    plt.plot(t, R, color="blue",label="Recovered",lw = 2,figure=fig)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Individuals in State")
    plt.title(title)
    plt.savefig(fname)
    return S, I, R

def random_SIR_simulation(j,N,n,k, network_create_func, network_set_func, direct_func, tMax, maxalpha, maxbeta, title, fname):
    '''
    Method to run and plot random network SIR experimentation
    Returns the median SIR values for parsed direct method
    '''
    S = {}
    I = {}
    R = {}
    fig = plt.figure()
    t = np.linspace(0,tMax,1000)
    num = N*n
    for i in range(j):
        network = network_create_func(N,n,k)
        iTotal, sTotal, rTotal, numInfNei, numSusNei, susceptible, infecteds = network_set_func(network,N,n)
        rates = np.zeros(2*num)
        for inf in infecteds:
            # set recovery hazard
            rates[num+inf] = maxalpha
            # set infection hazard
            rates[inf] = maxbeta*numSusNei[inf]/network.degree(inf)
        ti, Si, Ii, Ri = direct_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        S[i] = np.interp(t, ti, Si, right=Si[-1])
        I[i] = np.interp(t, ti, Ii, right=Ii[-1])
        R[i] = np.interp(t, ti, Ri, right=Ri[-1])
        plt.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
        plt.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
        plt.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
    S = np.median(np.array(list(S.values())),0)
    I = np.median(np.array(list(I.values())),0)
    R = np.median(np.array(list(R.values())),0)
    plt.plot(t, S, color="green",label="Susceptible",lw = 2,figure=fig)
    plt.plot(t, I, color="red",label="Infected",lw = 2,figure=fig)
    plt.plot(t, R, color="blue",label="Recovered",lw = 2,figure=fig)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Individuals in State")
    plt.title(title)
    plt.savefig(fname)
    return S, I, R

def nM_SIR_simulation(k, m_func, nm_func, haz_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta, haz_params, haz_max, title, fname):
    '''
    Method to run and plot Markov. vs. NonMarkov. experimentation
    Returns the median SIR values for the Markov. and NonMarkov. experiments
    '''
    t = np.linspace(0,tMax,1000)
    Sm = {}
    Im = {}
    Rm = {}
    Snm = {}
    Inm = {}
    Rnm = {}
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    for i in range(k):
        ti, Si, Ii, Ri = m_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        Sm[i] = np.interp(t, ti, Si, right=Si[-1])
        Im[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rm[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax1.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax1.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax1.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        ti, Si, Ii, Ri = nm_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(susceptible), haz_params, haz_func, haz_max)
        Snm[i] = np.interp(t, ti, Si, right=Si[-1])
        Inm[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rnm[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax2.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax2.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax2.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
    Sm = np.median(np.array(list(Sm.values())),0)
    Im = np.median(np.array(list(Im.values())),0)
    Rm = np.median(np.array(list(Rm.values())),0)
    Snm = np.median(np.array(list(Snm.values())),0)
    Inm = np.median(np.array(list(Inm.values())),0)
    Rnm = np.median(np.array(list(Rnm.values())),0)
    line_labels = ["Susceptible", "Infected", "Recovered"]
    S = ax1.plot(t, Sm, color="green",lw = 2)[0]
    I = ax1.plot(t, Im, color="red",lw = 2)[0]
    R = ax1.plot(t, Rm, color="blue",lw = 2)[0]
    ax2.plot(t, Snm, color="green",lw = 2)
    ax2.plot(t, Inm, color="red",lw = 2)
    ax2.plot(t, Rnm, color="blue",lw = 2)
    ax1.set_xlabel('Time')
    ax2.set_xlabel('Time')
    ax1.set_ylabel('Number of Individuals in State')
    fig.legend(handles=[S, I, R],labels=line_labels,loc="center right")
    fig.suptitle(title)
    plt.savefig(fname)
    return Sm,Im,Rm,Snm,Inm,Rnm