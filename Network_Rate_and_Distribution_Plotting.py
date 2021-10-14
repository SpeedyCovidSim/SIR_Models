import numpy as np
import igraph as ig
from matplotlib import pyplot as plt



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
