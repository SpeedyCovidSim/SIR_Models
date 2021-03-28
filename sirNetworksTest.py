import sirNetworksData as data
import sirNetworksAttr as attr
import numpy as np
import time
import igraph as ig

def main():
    '''
    Main loop for testing within this Python file
    '''
    # testing the gillespieDirectNetwork function
    # note random seed set within network model so result will occur everytime

    tMax = 200
    alpha = 0.4
    beta = 10 / 10000

    network = ig.Graph.Erdos_Renyi(10000,0.1)

    start = time.time()
    for i in range(10):
        print(f"Beginning simulation {i+1}")
        iTotal, sTotal, rTotal, numInfNei, rates, susceptible = data.setNetwork(network, alpha, beta)
        t, S, I, R = data.gillespieDirectNetwork(tMax, network, iTotal, sTotal, rTotal, numInfNei, rates, susceptible, alpha, beta)
    end = time.time()

    print(f"Avg. time taken for data sims: {(end-start)/10}")

    start = time.time()
    for i in range(10):
        print(f"Beginning simulation {i+1}")
        network = attr.setNetwork(network, alpha, beta)
        t, S, I, R = attr.gillespieDirectNetwork(tMax, network)
    end = time.time()

    print(f"Avg. time taken for attr sims: {(end-start)/10}")

if __name__=="__main__":
    main()