import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import igraph as ig

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
    plt.close()
    return Sfreq,Ifreq,Rfreq,Sdens,Idens,Rdens

def freq_dens_simulation_prop(k, freq_func, dens_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta, title, fname):
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
    N = network.vcount()
    for i in range(k):
        ti, Si, Ii, Ri = freq_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        Sfreq[i] = np.interp(t, ti, Si, right=Si[-1])/N
        Ifreq[i] = np.interp(t, ti, Ii, right=Ii[-1])/N
        Rfreq[i] = np.interp(t, ti, Ri, right=Ri[-1])/N
        ax1.plot(ti, Si/N, color="#82c7a5",lw = 2, alpha=0.3)
        ax1.plot(ti, Ii/N, color="#f15e22",lw = 2, alpha=0.3)
        ax1.plot(ti, Ri/N, color="#7890cd",lw = 2, alpha=0.3)
        ti, Si, Ii, Ri = dens_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        Sdens[i] = np.interp(t, ti, Si, right=Si[-1])/N
        Idens[i] = np.interp(t, ti, Ii, right=Ii[-1])/N
        Rdens[i] = np.interp(t, ti, Ri, right=Ri[-1])/N
        ax2.plot(ti, Si/N, color="#82c7a5",lw = 2, alpha=0.3)
        ax2.plot(ti, Ii/N, color="#f15e22",lw = 2, alpha=0.3)
        ax2.plot(ti, Ri/N, color="#7890cd",lw = 2, alpha=0.3)
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
    ax1.set_ylabel('Proportion of Individuals in State')
    fig.legend(handles=[S, I, R],labels=line_labels,loc="center right")
    fig.suptitle(title)
    plt.savefig(fname)
    plt.close()
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
    plt.close()
    return S, I, R

def general_proportional_SIR_simulation(k, direct_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta, title, fname):
    '''
    Method to run and plot general SIR experimentation
    Returns the median SIR values for parsed direct method
    '''
    S = {}
    I = {}
    R = {}
    fig = plt.figure()
    t = np.linspace(0,tMax,1000)
    N = network.vcount()
    for i in range(k):
        ti, Si, Ii, Ri = direct_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        S[i] = np.interp(t, ti, Si, right=Si[-1])/N
        I[i] = np.interp(t, ti, Ii, right=Ii[-1])/N
        R[i] = np.interp(t, ti, Ri, right=Ri[-1])/N
        plt.plot(ti, Si/N, color="#82c7a5",lw = 2, alpha=0.3,figure=fig)
        plt.plot(ti, Ii/N, color="#f15e22",lw = 2, alpha=0.3,figure=fig)
        plt.plot(ti, Ri/N, color="#7890cd",lw = 2, alpha=0.3,figure=fig)
    S = np.median(np.array(list(S.values())),0)
    I = np.median(np.array(list(I.values())),0)
    R = np.median(np.array(list(R.values())),0)
    plt.plot(t, S, color="green",label="Susceptible/N",lw = 2,figure=fig)
    plt.plot(t, I, color="red",label="Infected/N",lw = 2,figure=fig)
    plt.plot(t, R, color="blue",label="Recovered/N",lw = 2,figure=fig)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion of Individuals in State")
    plt.title(title)
    plt.savefig(fname)
    plt.close()
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
    plt.close()
    return S, I, R

def nM_SIR_simulation(k, m_func, nm_func, haz_func, tMax, network, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2):
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
    Snms = {}
    Inms = {}
    Rnms = {}
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    fig2, (ax3,ax4) = plt.subplots(nrows=1, ncols=2, sharex=True)
    for i in range(k):
        ti, Si, Ii, Ri = m_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta)
        Sm[i] = np.interp(t, ti, Si, right=Si[-1])
        Im[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rm[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax1.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax1.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax1.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        ax3.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax3.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax3.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        ti, Si, Ii, Ri = nm_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(susceptible), haz_params, haz_func, haz_max)
        Snm[i] = np.interp(t, ti, Si, right=Si[-1])
        Inm[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rnm[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax2.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax2.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax2.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        ti, Si, Ii, Ri = nm_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(susceptible), haz_params_strong, haz_func, haz_max_strong)
        Snms[i] = np.interp(t, ti, Si, right=Si[-1])
        Inms[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rnms[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax4.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax4.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax4.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)

    Sm = np.median(np.array(list(Sm.values())),0)
    Im = np.median(np.array(list(Im.values())),0)
    Rm = np.median(np.array(list(Rm.values())),0)
    Snm = np.median(np.array(list(Snm.values())),0)
    Inm = np.median(np.array(list(Inm.values())),0)
    Rnm = np.median(np.array(list(Rnm.values())),0)
    Snms = np.median(np.array(list(Snms.values())),0)
    Inms = np.median(np.array(list(Inms.values())),0)
    Rnms = np.median(np.array(list(Rnms.values())),0)
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
    fig.savefig(fname)
    plt.close(fig)
    S = ax3.plot(t, Sm, color="green",lw = 2)[0]
    I = ax3.plot(t, Im, color="red",lw = 2)[0]
    R = ax3.plot(t, Rm, color="blue",lw = 2)[0]
    ax4.plot(t, Snms, color="green",lw = 2)
    ax4.plot(t, Inms, color="red",lw = 2)
    ax4.plot(t, Rnms, color="blue",lw = 2)
    ax3.set_xlabel('Time')
    ax4.set_xlabel('Time')
    ax3.set_ylabel('Number of Individuals in State')
    fig2.legend(handles=[S, I, R],labels=line_labels,loc="center right")
    fig2.suptitle(title)
    fig2.savefig(fname2)
    plt.close(fig2)

    return Sm,Im,Rm,Snm,Inm,Rnm,Snms,Inms,Rnms

def nM_SEIR_simulation(k, m_func, nm_func, haz_func, tMax, network, eTotal, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta, maxgamma, haz_params, haz_max, haz_params_strong, haz_max_strong, title, fname, fname2):
    '''
    Method to run and plot Markov. vs. NonMarkov. experimentation
    Returns the median SIR values for the Markov. and NonMarkov. experiments
    '''
    t = np.linspace(0,tMax,1000)
    Sm = {}
    Em = {}
    Im = {}
    Infm = {}
    Rm = {}
    Snm = {}
    Inm = {}
    Rnm = {}
    Snms = {}
    Inms = {}
    Rnms = {}
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    fig2, (ax3,ax4) = plt.subplots(nrows=1, ncols=2, sharex=True)
    for i in range(k):
        ti, Si, Ei, Ii, Ri = m_func(tMax, network, eTotal, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(rates), copy.copy(susceptible), maxalpha, maxbeta, maxgamma)
        Sm[i] = np.interp(t, ti, Si, right=Si[-1])
        Em[i] = np.interp(t, ti, Ei, right=Ei[-1])
        Im[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rm[i] = np.interp(t, ti, Ri, right=Ri[-1])
        Inmi = Ei + Ii
        Infm[i] = Em[i] + Im[i]
        ax1.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax1.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax1.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        ax1.plot(ti, Ei, color="orange",lw = 2, alpha=0.3)
        ax1.plot(ti, Inmi, color="purple",lw=2, alpha=0.3)
        ax3.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax3.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax3.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        ax3.plot(ti, Ei, color="orange",lw = 2, alpha=0.3)
        ax3.plot(ti, Inmi, color="purple",lw=2, alpha=0.3)
        ti, Si, Ii, Ri = nm_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(susceptible), haz_params, haz_func, haz_max)
        Snm[i] = np.interp(t, ti, Si, right=Si[-1])
        Inm[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rnm[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax2.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax2.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax2.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
        ti, Si, Ii, Ri = nm_func(tMax, network, iTotal, sTotal, rTotal, copy.copy(numSusNei), copy.copy(susceptible), haz_params_strong, haz_func, haz_max_strong)
        Snms[i] = np.interp(t, ti, Si, right=Si[-1])
        Inms[i] = np.interp(t, ti, Ii, right=Ii[-1])
        Rnms[i] = np.interp(t, ti, Ri, right=Ri[-1])
        ax4.plot(ti, Si, color="#82c7a5",lw = 2, alpha=0.3)
        ax4.plot(ti, Ii, color="#f15e22",lw = 2, alpha=0.3)
        ax4.plot(ti, Ri, color="#7890cd",lw = 2, alpha=0.3)
    
    Sm = np.median(np.array(list(Sm.values())),0)
    Em = np.median(np.array(list(Em.values())),0)
    Im = np.median(np.array(list(Im.values())),0)
    Infm = np.median(np.array(list(Infm.values())),0)
    Rm = np.median(np.array(list(Rm.values())),0)
    Snm = np.median(np.array(list(Snm.values())),0)
    Inm = np.median(np.array(list(Inm.values())),0)
    Rnm = np.median(np.array(list(Rnm.values())),0)
    Snms = np.median(np.array(list(Snms.values())),0)
    Inms = np.median(np.array(list(Inms.values())),0)
    Rnms = np.median(np.array(list(Rnms.values())),0)
    maes = np.mean(np.abs(Infm-Inms))
    maew = np.mean(np.abs(Infm-Inm))
    line_labels = ["Susceptible", "Exposed", "Infectious", "Infected", "Recovered"]
    S = ax1.plot(t, Sm, color="green",lw = 2)[0]
    E = ax1.plot(t, Em, color="orange",lw = 2)[0]
    I = ax1.plot(t, Im, color="red",lw = 2)[0]
    Infi = ax1.plot(t, Infm, color="purple",lw = 2)[0]
    R = ax1.plot(t, Rm, color="blue",lw = 2)[0]
    ax2.plot(t, Snm, color="green",lw = 2)
    ax2.plot(t, Inm, color="red",lw = 2)
    ax2.plot(t, Rnm, color="blue",lw = 2)
    ax1.set_xlabel('Time')
    ax2.set_xlabel('Time')
    ax1.set_ylabel('Number of Individuals in State')
    fig.legend(handles=[S, E, I, Infi, R],labels=line_labels,loc="center right")
    fig.suptitle(title)
    fig.savefig(fname)
    plt.close(fig)
    S = ax3.plot(t, Sm, color="green",lw = 2)[0]
    E = ax3.plot(t, Em, color="orange",lw = 2)[0]
    I = ax3.plot(t, Im, color="red",lw = 2)[0]
    Infi = ax3.plot(t, Infm, color="purple",lw = 2)[0]
    R = ax3.plot(t, Rm, color="blue",lw = 2)[0]
    ax4.plot(t, Snms, color="green",lw = 2)
    ax4.plot(t, Inms, color="red",lw = 2)
    ax4.plot(t, Rnms, color="blue",lw = 2)
    ax3.set_xlabel('Time')
    ax4.set_xlabel('Time')
    ax3.set_ylabel('Number of Individuals in State')
    fig2.legend(handles=[S, E, I, Infi, R],labels=line_labels,loc="center right")
    fig2.suptitle(title)
    fig2.savefig(fname2)
    plt.close(fig2)
    print(f"mae Weak nMarkov: {maew:.4f}")
    print(f"mae Strong nMarkov: {maes:.4f}")
    return Sm,Em,Im,Infm,Rm,Snm,Inm,Rnm,Snms,Inms,Rnms

def nH_single_simulation(k, update_func, rate_func, inv_method, thin_method, haz_max, tMax, title, fname, fname2):
    '''
    Method to run and plot nonhomogeneous Poisson validation experiments
    '''
    t = np.linspace(0,tMax,1000)
    invn = {}
    thinn = {}
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    for i in range(k):
        ti, ei = inv_method(update_func, tMax)
        ei = np.cumsum(ei)
        invn[i] = np.interp(t, ti, ei,right=ei[-1])
        ax1.plot(ti, ei, color="#82c7a5",lw = 2, alpha=0.3)
        ti, ei = thin_method(rate_func, haz_max, tMax)
        ti = np.append(ti, tMax)
        ei = np.append(ei, ei[-1])
        ei = np.cumsum(ei)
        thinn[i] = np.interp(t, ti, ei,right=ei[-1])
        ax2.plot(ti, ei, color="#7890cd",lw = 2, alpha=0.3)
    Invm = np.median(np.array(list(invn.values())),0)
    Thinm = np.median(np.array(list(thinn.values())),0)
    inv = ax1.plot(t, Invm, color="green",lw = 2)[0]
    thin = ax2.plot(t, Thinm, color="blue",lw = 2)[0]
    ax1.plot(t, Thinm, color="blue",linestyle="dashed", lw=2,alpha=0.7)
    ax2.plot(t, Invm, color="green",lw=2,linestyle="dashed",alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_xlim([0,50])
    ax2.set_xlim([0,50])
    ax2.set_xlabel('Time')
    ax1.set_ylabel('Cumulative Number of Events')
    line_labels = ["Inverse Method", "Thinning Method"]
    fig.legend(handles=[inv, thin],labels=line_labels,loc="center right")
    fig.suptitle(title)
    plt.savefig(fname)
    plt.close()

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    ax1.hist(np.array(list(invn.values()))[:,600])
    ax1.set_xlabel('Cumulative Number of Events')
    ax2.set_xlabel('Cumulative Number of Events')
    ax1.set_ylabel('Frequency')
    ax2.hist(np.array(list(thinn.values()))[:,600])
    plt.savefig(fname2)
    plt.close()


def nH_competing_simulation(k, update_func, rate_func_sing, rate_func_vec, inv_method, nMGA, first_method, gMax, haz_max, haz_min, num_processes, tMax, title, fname):
    '''
    Method to run and plot nonhomogeneous Poisson validation experiments
    '''
    t = np.linspace(0,tMax,1000)
    inv0m = {}
    inv1m = {}
    inv2m = {}
    inv3m = {}
    nMGA0m = {}
    nMGA1m = {}
    nMGA2m = {}
    nMGA3m = {}
    first0m = {}
    first1m = {}
    first2m = {}
    first3m = {}
    max0m = {}
    max1m = {}
    max2m = {}
    max3m = {}
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]
    for i in range(k):
        ti, invReactTypes = inv_method(update_func, num_processes, tMax)
        inv0 = np.cumsum(invReactTypes==0)
        inv1 = np.cumsum(invReactTypes==1)
        inv2 = np.cumsum(invReactTypes==2)
        inv3 = np.cumsum(invReactTypes==3)
        inv0m[i] = np.interp(t, ti, inv0)
        inv1m[i] = np.interp(t, ti, inv1)
        inv2m[i] = np.interp(t, ti, inv2)
        inv3m[i] = np.interp(t, ti, inv3)
        ax1.plot(ti, inv0, label="Event Type 0",color="#82c7a5",lw = 2, alpha=0.15)
        ax1.plot(ti, inv1, label="Event Type 1",color="#f15e22",lw = 2, alpha=0.15)
        ax1.plot(ti, inv2, label="Event Type 2",color="#7890cd",lw = 2, alpha=0.15)
        ax1.plot(ti, inv3, label="Event Type 3",color="purple",lw = 2, alpha=0.15)
        ax1.set_title("Inverse")
        ti, firstMaxReactTypes = first_method(rate_func_sing, haz_max, tMax)
        first0 = np.cumsum(firstMaxReactTypes==0)
        first1 = np.cumsum(firstMaxReactTypes==1)
        first2 = np.cumsum(firstMaxReactTypes==2)
        first3 = np.cumsum(firstMaxReactTypes==3)
        first0m[i] = np.interp(t, ti, first0)
        first1m[i] = np.interp(t, ti, first1)
        first2m[i] = np.interp(t, ti, first2)
        first3m[i] = np.interp(t, ti, first3)
        ax2.plot(ti, first0, label="Event Type 0",color="#82c7a5",lw = 2, alpha=0.15)
        ax2.plot(ti, first1, label="Event Type 1",color="#f15e22",lw = 2, alpha=0.15)
        ax2.plot(ti, first2, label="Event Type 2",color="#7890cd",lw = 2, alpha=0.15)
        ax2.plot(ti, first3, label="Event Type 3",color="purple",lw = 2, alpha=0.15)
        ax2.set_title("First Max")
        ti, MaxReactTypes = gMax(rate_func_sing, haz_max, tMax)
        max0 = np.cumsum(MaxReactTypes==0)
        max1 = np.cumsum(MaxReactTypes==1)
        max2 = np.cumsum(MaxReactTypes==2)
        max3 = np.cumsum(MaxReactTypes==3)
        max0m[i] = np.interp(t, ti, max0)
        max1m[i] = np.interp(t, ti, max1)
        max2m[i] = np.interp(t, ti, max2)
        max3m[i] = np.interp(t, ti, max3)
        ax3.plot(ti, max0, label="Event Type 0",color="#82c7a5",lw = 2, alpha=0.15)
        ax3.plot(ti, max1, label="Event Type 1",color="#f15e22",lw = 2, alpha=0.15)
        ax3.plot(ti, max2, label="Event Type 2",color="#7890cd",lw = 2, alpha=0.15)
        ax3.plot(ti, max3, label="Event Type 3",color="purple",lw = 2, alpha=0.15)
        ax3.set_title("GMax")
        ti, nMGAReactTypes = nMGA(rate_func_vec, haz_min, tMax)
        nMGA0 = np.cumsum(nMGAReactTypes==0)
        nMGA1 = np.cumsum(nMGAReactTypes==1)
        nMGA2 = np.cumsum(nMGAReactTypes==2)
        nMGA3 = np.cumsum(nMGAReactTypes==3)
        nMGA0m[i] = np.interp(t, ti, nMGA0)
        nMGA1m[i] = np.interp(t, ti, nMGA1)
        nMGA2m[i] = np.interp(t, ti, nMGA2)
        nMGA3m[i] = np.interp(t, ti, nMGA3)
        ax4.plot(ti, nMGA0, label="Event Type 0",color="#82c7a5",lw = 2, alpha=0.15)
        ax4.plot(ti, nMGA1, label="Event Type 1",color="#f15e22",lw = 2, alpha=0.15)
        ax4.plot(ti, nMGA2, label="Event Type 2",color="#7890cd",lw = 2, alpha=0.15)
        ax4.plot(ti, nMGA3, label="Event Type 3",color="purple",lw = 2, alpha=0.15)
        ax4.set_title("nMGA")
    inv0m = np.median(np.array(list(inv0m.values())),0)
    inv1m = np.median(np.array(list(inv1m.values())),0)
    inv2m = np.median(np.array(list(inv2m.values())),0)
    inv3m = np.median(np.array(list(inv3m.values())),0)
    first0m = np.median(np.array(list(first0m.values())),0)
    first1m = np.median(np.array(list(first1m.values())),0)
    first2m = np.median(np.array(list(first2m.values())),0)
    first3m = np.median(np.array(list(first3m.values())),0)
    max0m = np.median(np.array(list(max0m.values())),0)
    max1m = np.median(np.array(list(max1m.values())),0)
    max2m = np.median(np.array(list(max2m.values())),0)
    max3m = np.median(np.array(list(max3m.values())),0)
    nMGA0m = np.median(np.array(list(nMGA0m.values())),0)
    nMGA1m = np.median(np.array(list(nMGA1m.values())),0)
    nMGA2m = np.median(np.array(list(nMGA2m.values())),0)
    nMGA3m = np.median(np.array(list(nMGA3m.values())),0)

    p1 = ax1.plot(t, inv0m, color="green",lw = 2)[0]
    p2 = ax1.plot(t, inv1m, color="red",lw = 2)[0]
    p3 = ax1.plot(t, inv2m, color="blue",lw = 2)[0]
    p4 = ax1.plot(t, inv3m, color="purple",lw = 2)[0]
    ax2.plot(t, first0m, color="green",lw = 2)
    ax2.plot(t, first1m, color="red",lw = 2)
    ax2.plot(t, first2m, color="blue",lw = 2)
    ax2.plot(t, first3m, color="purple",lw = 2)
    ax3.plot(t, max0m, color="green",lw = 2)
    ax3.plot(t, max1m, color="red",lw = 2)
    ax3.plot(t, max2m, color="blue",lw = 2)
    ax3.plot(t, max3m, color="purple",lw = 2)
    ax4.plot(t, nMGA0m, color="green",lw = 2)
    ax4.plot(t, nMGA1m, color="red",lw = 2)
    ax4.plot(t, nMGA2m, color="blue",lw = 2)
    ax4.plot(t, nMGA3m, color="purple",lw = 2)
    ax3.set_xlabel('Time')
    ax4.set_xlabel('Time')
    fig.text(0.04, 0.5, 'Cumulative Number of Events Occurred', va='center', rotation='vertical')
    line_labels = ["Event Type 0", "Event Type 1", "Event Type 2", "Event Type 3"]
    plt.legend(handles=[p1, p2, p3, p4],labels=line_labels,bbox_to_anchor=(1.04,1.5), loc="upper left")

    plt.savefig(fname,bbox_inches="tight")
    plt.close()

def nH_timing_simulation(k, update_funcs, rate_func_sings, rate_func_vecs, inv_method, nMGA, first_method, gMax, haz_maxs, haz_mins, num_processes, tMax):
    times = {}
    times["Inverse"] = np.zeros(len(update_funcs))
    times["First"] = np.zeros(len(rate_func_sings))
    times["GMax"] = np.zeros(len(rate_func_sings))
    times["nMGA"] = np.zeros(len(rate_func_vecs))

    for i in range(len(update_funcs)):
        s_time = time.time()
        for j in range(k):
            inv_method(update_funcs[i], num_processes[i], tMax)
        e_time = time.time()
        times["Inverse"][i] = (e_time-s_time)
    
    for i in range(len(rate_func_sings)):
        s_time = time.time()
        for j in range(k):
            first_method(rate_func_sings[i], haz_maxs[i], tMax)
        e_time = time.time()
        times["First"][i] = (e_time-s_time)

    for i in range(len(rate_func_sings)):
        s_time = time.time()
        for j in range(k):
            gMax(rate_func_sings[i], haz_maxs[i], tMax)
        e_time = time.time()
        times["GMax"][i] = (e_time-s_time)

    for i in range(len(rate_func_vecs)):
        s_time = time.time()
        for j in range(k):
            nMGA(rate_func_vecs[i], haz_mins[i], tMax)
        e_time = time.time()
        times["nMGA"][i] = (e_time-s_time)
    
    for ke, v in times.items():
        print(ke, v)

def ode_N_simulation(g_func, f_func, tMax, networks, iTotal, sTotal, rTotal, numSusNei, rates, susceptible, maxalpha, maxbeta, title, fname):
    '''
    Method to run and plot freq. vs. dens. experimentation
    Returns the median SIR values for freq. and dens. scalings 
    '''
    t = np.linspace(0,tMax,1000)


    network = networks[0]
    network1 = networks[1]
    rates1 = rates[1]
    rates = rates[0]

    fig, axs = plt.subplots(nrows=2, ncols=2)
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]

    t, Si, Ii, Ri = g_func(tMax, network, iTotal[0], sTotal[0], rTotal[0], copy.copy(numSusNei[0]), copy.copy(rates), copy.copy(susceptible[0]), maxalpha, maxbeta)
    Si = np.append(Si,Si[-1])
    t = np.append(t,tMax)
    Ii = np.append(Ii,Ii[-1])
    Ri = np.append(Ri,Ri[-1])
    S=ax1.plot(t, Si, color="green",lw = 2)[0]
    I=ax1.plot(t, Ii, color="red",lw = 2)[0]
    R=ax1.plot(t, Ri, color="blue",lw = 2)[0]
    line_labels = ["Susceptible", "Infected", "Recovered"]
    ax1.text(6,60,"N=100\nDirect Method")
    plt.legend(handles=[S, I, R],labels=line_labels,bbox_to_anchor=(1.04,1.5), loc="upper left")
    t, Si, Ii, Ri = g_func(tMax, network1, iTotal[1], sTotal[1], rTotal[1], copy.copy(numSusNei[1]), copy.copy(rates1), copy.copy(susceptible[1]), maxalpha, maxbeta)
    Si = np.append(Si,Si[-1])
    t = np.append(t,tMax)
    Ii = np.append(Ii,Ii[-1])
    Ri = np.append(Ri,Ri[-1])
    ax2.plot(t, Si, color="green",lw = 2)
    ax2.plot(t, Ii, color="red",lw = 2)
    ax2.plot(t, Ri, color="blue",lw = 2)
    t, Si, Ii, Ri = f_func(tMax, network, iTotal[0], sTotal[0], rTotal[0], copy.copy(numSusNei[0]), copy.copy(rates), copy.copy(susceptible[0]), maxalpha, maxbeta)
    Si = np.append(Si,Si[-1])
    t = np.append(t,tMax)
    Ii = np.append(Ii,Ii[-1])
    Ri = np.append(Ri,Ri[-1])
    ax3.plot(t, Si, color="green",lw = 2)
    ax3.plot(t, Ii, color="red",lw = 2)
    ax3.plot(t, Ri, color="blue",lw = 2)
    t, Si, Ii, Ri = f_func(tMax, network1, iTotal[1], sTotal[1], rTotal[1], copy.copy(numSusNei[1]), copy.copy(rates1), copy.copy(susceptible[1]), maxalpha, maxbeta)
    Si = np.append(Si,Si[-1])
    t = np.append(t,tMax)
    Ii = np.append(Ii,Ii[-1])
    Ri = np.append(Ri,Ri[-1])
    ax4.plot(t, Si, color="green",lw = 2)
    ax4.plot(t, Ii, color="red",lw = 2) 
    ax4.plot(t, Ri, color="blue",lw = 2)
    ax4.text(8,6000,"N=10000\nNext Reaction Method")
    ax3.set_xlabel('Time')
    ax4.set_xlabel('Time')
    

    fig.text(0.04, 0.5, 'Number of Individuals in State', va='center', rotation='vertical')
    fig.suptitle(title)
    plt.savefig(fname,bbox_inches="tight")
    plt.close()