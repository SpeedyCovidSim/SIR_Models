import numpy as np
import igraph as ig
from numpy import random
from plots import plotSIR, plotSIRK

def setNetwork(network, alpha, beta, prop_i=0.05):
    '''
    Initialises a parsed network with required random infected individuals (1 if too small)
    Inputs
    network     : igraph network object
    prop_i      : proportion of population that is infected (default 0.05)
    Outputs
    network     : formatted network (note this may need to be edited due to Python's annoying object copying system)
    infecteds   : list of infected vertex IDs
    '''
    # get random proportion of populace to be infected
    N = network.vcount()
    num_infected = int(np.ceil(prop_i*N))
    infecteds = list(random.choice(N, num_infected, replace=False))

    # set SIR numbers
    network["I_total"] = num_infected
    network["S_total"] = N-num_infected
    network["R_total"] = 0

    # set disease params
    network["alpha"] = alpha
    network["beta"] = beta

    # set states of individuals according to random draws
    network.vs["state"]="S"
    network.vs[infecteds]["state"]="I"

    # adding in hazards
    network = initHazards(network)
    return network

def initHazards(network):
    # init num_inf_nei as 0 before looping
    network.vs["num_inf_nei"] = 0
    # loop over all infected vertices
    for inf_vert in network.vs(state_eq="I"):
        # LHS: array of num_inf_nei for neighbouring vertices
        # RHS: list(add 1 to array of num_inf_nei for neighbouring vertices), note casting is required for easy array operations
        network.vs[network.neighbors(inf_vert)]["num_inf_nei"]=list(np.array(network.vs[network.neighbors(inf_vert)]["num_inf_nei"])+1)
        # add in recovery
        inf_vert["rate"] = network["alpha"]
    # for all susceptible vertices add in infection hazard
    # LHS: array of hazards for susceptible vertices
    # RHS: hazards calculated as beta*(num of infected neighbours), note casting is required for easy array operations
    network.vs(state_eq="S")["rate"] = list(np.array(network.vs(state_eq="S")["num_inf_nei"])*network["beta"])
    return network

def gillespieDirectNetwork(t_max, network, t_init = 0.0):
        '''
        Direct Gillespie Method, on network
        Uses numpy's random module for r.v. and sampling
        Inputs
        t_init  : Initial time (default of 0)
        t_max   : Simulation end time
        network : population network
        alpha   : probability of infected person recovering [0,1]
        beta    : probability of susceptible person being infected [0,1]
        Outputs
        t       : Array of times at which events have occured
        S       : Array of num people susceptible at each t
        I       : Array of num people infected at each t
        R       : Array of num people recovered at each t
        '''

        # initialise outputs and preallocate space
        maxi = network.vcount()*4
        S = np.zeros(maxi)
        t = 1*S
        t[0] = t_init
        I = 1*S
        R = 1*S
        S[0] = network["S_total"]
        I[0] = network["I_total"]
        R[0] = network["R_total"]
        i = 1


        # initialise random variate generation with set seed
        rng = random.default_rng(123)

        while t[-1] < t_max and network["I_total"] != 0:
            # get hazard sum and next event time
            h = np.sum(network.vs["rate"])
            delta_t = rng.exponential(1/h)

            # choose the index of the individual to transition
            eventIndex = random.choice(a=network.vcount(),p=(network.vs["rate"]/np.sum(network.vs["rate"])))

            # update local neighbourhood attributes
            if network.vs[eventIndex]["state"] == "S":  # (S->I)
                # change state and individual rate
                network.vs[eventIndex]["state"] = "I"
                network.vs[eventIndex]["rate"] = network["alpha"]
                # increase num_inf_neighbours for neighbouring vertices
                network.vs[network.neighbors(eventIndex)]["num_inf_nei"] = list(np.array(network.vs[network.neighbors(eventIndex)]["num_inf_nei"])+1)
                # update hazards of neighbouring susceptible vertices
                network.vs[network.neighbors(eventIndex)](state_eq='S')["rate"] = list(np.array(network.vs[network.neighbors(eventIndex)](state_eq='S')["num_inf_nei"])*network["beta"])
                # update network totals
                network["S_total"] -= 1
                network["I_total"] += 1

            elif network.vs[eventIndex]["state"] == "I": # (I->R)
                # change state and individual rate
                network.vs[eventIndex]["state"] = "R"
                network.vs[eventIndex]["rate"] = 0
                # decrease num_inf_neighbours for neighbouring vertices
                network.vs[network.neighbors(eventIndex)]["num_inf_nei"] = list(np.array(network.vs[network.neighbors(eventIndex)]["num_inf_nei"])-1)
                # update hazards of neighbouring susceptible vertices
                network.vs[network.neighbors(eventIndex)](state_eq='S')["rate"] = list(np.array(network.vs[network.neighbors(eventIndex)](state_eq='S')["num_inf_nei"])*network["beta"])
                # update network totals
                network["I_total"] -= 1
                network["R_total"] += 1

            if i < maxi:
                t[i] = t[i-1] + delta_t
                S[i] = network["S_total"]
                I[i] = network["I_total"]
                R[i] = network["R_total"]
            else:
                t.append(t[1] + delta_t)
                S.append(network["S_total"])
                I.append(network["I_total"])
                R.append(network["R_total"])

            i += 1
    
        S = S[:i]
        t = t[:i]
        R = R[:i]
        I = I[:i]
        return t, S, I, R

def main():
    '''
    Main loop for testing within this Python file
    '''
    # testing the gillespieDirect2Processes function
    # note random seed set within network model so result will occur everytime

    # initialise variables
    # N = np.array([5, 10, 50, 100,1000])
    # k = [2,3,10,20,100]
    N = np.array([5, 10, 50, 100,1000,10000])
    k = [2,3,10,20,100,1000]


    t_max = 200
    alpha = 0.4
    beta = 10 / N

    # iterate through populations for complete graphs
    if True:
        print("Beginning full graph simulations")
        for i in range(len(N)):
            print(f"Iteration {i} commencing")
            network = ig.Graph.Full(N[i])
            network = setNetwork(network, alpha, beta[i])
            print(f"Beginning simulation {i}")
            t, S, I, R = gillespieDirectNetwork(t_max, network)
            print(f"Exporting simulation {i}")
            # plot and export the simulation
            outputFileName = f"pythonGraphs/SpeedTest/networkDirectSIR/SIR_Model_Pop_{N[i]}"
            plotSIR(t, [S, I, R], alpha, beta[i], N[i], outputFileName, Display=False)

    if True:   
        print("Beginning connectedness simulations")
        for i in range(len(N)):
            print(f"Iteration {i} commencing")
            network = ig.Graph.K_Regular(N[i], k[i])
            network = setNetwork(network, alpha, beta[i])
            print(f"Beginning simulation {i}")
            t, S, I, R = gillespieDirectNetwork(t_max, network)
            print(f"Exporting simulation {i}")
            # plot and export the simulation
            outputFileName = f"pythonGraphs/SpeedTest/networkDirectDegreeSIR/SIR_Model_Pop_{N[i]}"
            plotSIRK(t, [S, I, R], alpha, beta[i], N[i], k[i], outputFileName, Display=False)

if __name__=="__main__":
    main()