import sirNetworksData as data
import sirNetworksAttr as attr
import sirNetworksNextReact as next
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

    if False:
        start = time.time()
        for i in range(1000):
            neighbors = network.neighbors(100)
        end = time.time()

        print(f"Avg. time taken for inbuilt method: {(end-start)/1000}")

        start = time.time()
        for i in range(10):
            neighbors = {}
            for i in range(10000):
                neighbors[i] = network.neighbors(i)
        end = time.time()

        print(f"Avg. time taken to build dictionary: {(end-start)/10}")
        
        start = time.time()
        for i in range(1000):
            neighbor = neighbors[100] 
        end = time.time()

        print(f"Avg. time taken for dictionary method: {(end-start)/1000}")
        


    if True:
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

    if True:
        start = time.time()
        for i in range(10):
            print(f"Beginning simulation {i+1}")
            iTotal, sTotal, rTotal, numInfNei, rates, susceptible = next.setNetwork(network, alpha, beta)
            t, S, I, R = next.nextReactNetwork(tMax, network, iTotal, sTotal, rTotal, numInfNei, rates, susceptible, alpha, beta)
        end = time.time()

        print(f"Avg. time taken for data sims: {(end-start)/10}")

    if False:
        source = np.array([0,2,3,1,4,5] * 100000000)
        lookup_table = np.array([11,22,33,44,55,66])
        result = np.zeros_like(source)
        start = time.time()
        result = lookup_table.ravel()[source.ravel()]
        end = time.time()
        print(end -start)

        source2 = np.array([0,2,3,1,4,5] * 100000000)
        lookup_table2 = np.array([11,22,33,44,55,66])
        result2 = np.zeros_like(source)
        start = time.time()
        result2 = lookup_table2[source2]
        end = time.time()
        print(end -start)

    if False:
        x = np.random.rand(10000)
        start = time.time()
        for i in range(5000):
            rs = np.random.uniform(size=10000)
            tau = 1/x*np.log(1/rs)
        end = time.time()
        print(end - start)
        start = time.time()
        for i in range(5000):
            tau = np.random.exponential(x)
        end = time.time()
        print(end - start)

if __name__=="__main__":
    main()