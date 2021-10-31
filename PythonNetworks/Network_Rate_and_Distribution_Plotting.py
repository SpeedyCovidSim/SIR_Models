import numpy as np
from numpy import random
import igraph as ig
from matplotlib import pyplot as plt

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
    

    selections = np.arange(N)
    choices = []
    i = 0
    while i < k:
        houses = random.choice(selections,size=2,replace=False)
        if not(list(houses) in choices):
            network.add_edge(houses[0]*n,houses[1]*n+n-1)
            choices.append(list(houses))
            i += 1
    
    return network



def main(networks=True,rates=True,distributions=True):
    '''
    Main loop for testing within this Python file
    '''
    if(networks):
        network = ig.Graph.Watts_Strogatz(1,20,1,0)
        fig, ax = plt.subplots()
        fname = f"PythonPlotting/Misc/ws_1"
        ig.plot(network,target=ax)
        plt.axis('off')
        plt.savefig(fname)
        plt.close()

        network = ig.Graph.Watts_Strogatz(1,20,4,0)
        fig, ax = plt.subplots()
        fname = f"PythonPlotting/Misc/ws_4"
        ig.plot(network,target=ax)
        plt.axis('off')
        plt.savefig(fname)
        plt.close()

        network = create_linked_neighbourhood(10,4)
        fig, ax = plt.subplots()
        fname = f"PythonPlotting/Misc/household_lattice"
        ig.plot(network,target=ax)
        plt.axis('off')
        plt.savefig(fname)
        plt.close()

        network = create_random_neighbourhood(10,4,3)
        fig, ax = plt.subplots()
        fname = f"PythonPlotting/Misc/ws_households"
        ig.plot(network,target=ax)
        plt.axis('off')
        plt.savefig(fname)
        plt.close()

        network = create_random_household_links(10,4,3)
        fig, ax = plt.subplots()
        fname = f"PythonPlotting/Misc/er_households"
        ig.plot(network,target=ax)
        plt.axis('off')
        plt.savefig(fname)
        plt.close()

    if(rates):
        x = np.linspace(0,15,100)
        k1 = 1.1
        l1 = 0.2591
        k2 = 1.4
        l2 = 0.2743
        weib1 = k1/l1*(x/l1)**(k1-1)
        weib2 = k2/l2*(x/l2)**(k2-1)
        const = np.ones(len(x))*4
        fig,ax = plt.subplots()
        plt.plot(x,weib2,label="Strong Weibull Hazard")
        plt.plot(x,weib1,label="Weak Weibull Hazard")
        plt.plot(x,const,label="Markovian Hazard")
        plt.legend()
        plt.savefig("PythonPlotting/Misc/hazards")
        plt.close()

    if(distributions):
        x = np.linspace(0,10,100)
        k1 = 1.1
        l1 = 2.5909
        weib1 = k1/l1*(x/l1)**(k1-1)*np.exp(-(x/l1)**k1)
        const = 0.4*np.exp(-0.4*x)
        fig,ax = plt.subplots()
        plt.plot(x,weib1,label="Weibull Generation Time")
        plt.plot(x,const,label="Exponential Generation Time")
        plt.axvline(x=2.5,color="red",label="Mean Generation Time")
        plt.legend()
        plt.xlabel('Time (t)')
        plt.ylabel('P(t)')
        plt.savefig("PythonPlotting/Misc/recovery_dist")
        plt.close()

    


if __name__=="__main__":
    main()
