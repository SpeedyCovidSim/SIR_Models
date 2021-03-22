'''
This is a code base for the plotting function of a simple SIR simulation
using the Gillespie Direct Method

Author: Joel Trent and Josh Looker
'''
from matplotlib import pyplot as plt

def plotSIRD(t, SIRD, alpha, beta, N, outputFileName="plot", Display=True, save=True):
    '''
    Inputs
    t              : Array of times at which events have occured
    SIR            : Array of arrays of Num people susceptible, infected and recovered at
                     each t
    N              : Population size
    alpha          : probability of infected person recovering [0,1]
    beta           : probability of susceptible person being infected [0,1]
    outputFileName : the name/location to save the plot as/in

    Outputs
    png            : plot of SIR model over time [by default]
    '''
    fig = plt.figure()
    plt.plot(t, SIRD[0], label="Susceptible", lw = 2, figure=fig)
    plt.plot(t, SIRD[1], label="Infected", lw = 2, figure=fig)
    plt.plot(t, SIRD[2], label="Recovered", lw=2, figure=fig)
    plt.plot(t, SIRD[3], label="Dead", lw=2, figure=fig)

    plt.xlabel("Time")
    plt.ylabel("Population Number")
    plt.suptitle(f"SIRD model over time with a population size of {N}")
    plt.title(f"For alpha = {alpha} and beta {beta}")
    plt.legend()

    if Display:
        # required to display graph on plots.
        fig.show()

    if save:
        # Save graph as pngW
        fig.savefig(outputFileName)

def plotSIRDK(t, SIRD, alpha, beta, N, k, outputFileName="plot", Display=True, save=True):
    '''
    Inputs
    t              : Array of times at which events have occured
    SIR            : Array of arrays of Num people susceptible, infected and recovered at
                     each t
    N              : Population size
    k              : Degree of connectedness
    alpha          : probability of infected person recovering [0,1]
    beta           : probability of susceptible person being infected [0,1]
    outputFileName : the name/location to save the plot as/in

    Outputs
    png            : plot of SIR model over time [by default]
    '''
    fig = plt.figure()
    plt.plot(t, SIRD[0], label="Susceptible", lw = 2, figure=fig)
    plt.plot(t, SIRD[1], label="Infected", lw = 2, figure=fig)
    plt.plot(t, SIRD[2], label="Recovered", lw=2, figure=fig)
    plt.plot(t, SIRD[3], label="Dead", lw=2, figure=fig)

    plt.xlabel("Time")
    plt.ylabel("Population Number")
    plt.suptitle(f"SIRD model over time with a population size of {N} and connectedness {k}")
    plt.title(f"For alpha = {alpha} and beta {beta}")
    plt.legend()

    if Display:
        # required to display graph on plots.
        fig.show()

    if save:
        # Save graph as pngW
        fig.savefig(outputFileName)
