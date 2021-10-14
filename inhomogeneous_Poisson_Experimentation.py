import numpy as np
from matplotlib import pyplot as plt
from Python_Nonhomog.Single.nonHomogSim import inverseMethod as single_inverse, thinningApproxMethod as single_thin, thinningExactMethod as single_thin_exact
from Python_Nonhomog.Competing.nonHomogSim import firstInverseMethod as competing_inverse, firstThinningApproxMethod as competing_thin, gillespieMax as gMax, nMGA
from PythonSimulation.simulate import nH_single_simulation, nH_competing_simulation, nH_timing_simulation

def main(single=True, competing=True, timing=True):
    '''
    Main loop for testing within this Python file
    '''
    if (single):
        print("Single inhomogeneous Poisson experiments")
        # initialise variables and rate/update functions
        def updateFunction(ti, r):
            return np.sqrt((1+ti**2-r**(2/4))/(r**(2/4)))
        def rateFunction(t):
            return (4*t)/(1+t**2) 
        j = 30
        rateMax = 2
        tMax = 50
        # conducting experiment and plotting
        title = "Inversion and Thinning Methods for an inhomogeneous Poisson process"
        fname = "PythonPlotting/nh_Poisson_Tests/Single"
        nH_single_simulation(j, updateFunction, rateFunction, single_inverse, single_thin, rateMax, tMax, title, fname)


    if (competing):
        print("Competing inhomogeneous Poisson experiments")
        # initialise variables and rate/update functions
        def updateFunction(ti, u):
            tip1 = [
                np.sqrt((1+ti**2-u[0]**(2/4))/(u[0]**(2/4))), 
                np.sqrt((25+ti**2-25*u[1]**(2/20))/(u[1]**(2/20))),
                np.sqrt((81+ti**2-81*u[2]**(2/40))/(u[2]**(2/40))), 
                np.sqrt((169+ti**2-169*u[3]**(2/60))/(u[3]**(2/60)))]
            return tip1
        def rateFunctionSing(reactType, t): 
            if reactType == 0: 
                return (4*t)/(1+t**2) 
            elif reactType == 1: 
                return (20*t)/(25+t**2) 
            elif reactType == 2: 
                return (40*t)/(81+t**2) 
            else:
                return (60*t)/(169+t**2)
        def rateFunctionVect(t):
            rates = [(4*t)/(1+t**2),(20*t)/(25+t**2),(40*t)/(81+t**2),(60*t)/(169+t**2)]
            return rates
        j = 30

        # system params
        numProcesses = 4
        rateMax = np.array([2,2,2.23,2.31])
        minBounds = np.array([0.4,0.2,0.1,0.1])
        tMax = 40

        title = f"Inv., First Max., Gillespie Max., and nMGA Methods for competing \n inhomogeneous Poissons"
        fname= f"PythonPlotting/nh_Poisson_Tests/Competing"
        nH_competing_simulation(j, updateFunction, rateFunctionSing, rateFunctionVect, competing_inverse, nMGA, competing_thin, 
        gMax, rateMax, minBounds, numProcesses, tMax, title, fname)


    if (timing):
        print("Beginning timing experiments")
        # initialise variables and rate/update functions
        tMax = 50
        def updateFunctionOne(ti, r):
            return np.sqrt((1+ti**2-r**(2/4))/(r**(2/4)))
        def rateFunctionSingOne(reactType, t):
            return (4*t)/(1+t**2) 
        def rateFunctionVectOne(t):
            rates = [(4*t)/(1+t**2)]
            return rates
        j = 500
        rateMaxOne = np.array([2])
        minBoundsOne = np.array([0.4])

        
        def updateFunctionTwo(ti, u):
            tip1 = [
                np.sqrt((1+ti**2-u[0]**(2/4))/(u[0]**(2/4))), 
                np.sqrt((25+ti**2-25*u[1]**(2/20))/(u[1]**(2/20)))
            ]
            return tip1
        def rateFunctionSingTwo(reactType, t): 
            if reactType == 0: 
                return (4*t)/(1+t**2) 
            else: 
                return (20*t)/(25+t**2)
        def rateFunctionVectTwo(t):
            rates = [(4*t)/(1+t**2),(20*t)/(25+t**2)]
            return rates
        rateMaxTwo = np.array([2,2])
        minBoundsTwo = np.array([0.4,0.2])


        def updateFunctionFour(ti, u):
            tip1 = [
                np.sqrt((1+ti**2-u[0]**(2/4))/(u[0]**(2/4))), 
                np.sqrt((25+ti**2-25*u[1]**(2/20))/(u[1]**(2/20))),
                np.sqrt((81+ti**2-81*u[2]**(2/40))/(u[2]**(2/40))), 
                np.sqrt((169+ti**2-169*u[3]**(2/60))/(u[3]**(2/60)))]
            return tip1
        def rateFunctionSingFour(reactType, t): 
            if reactType == 0: 
                return (4*t)/(1+t**2) 
            elif reactType == 1: 
                return (20*t)/(25+t**2) 
            elif reactType == 2: 
                return (40*t)/(81+t**2) 
            else:
                return (60*t)/(169+t**2)
        def rateFunctionVectFour(t):
            rates = [(4*t)/(1+t**2),(20*t)/(25+t**2),(40*t)/(81+t**2),(60*t)/(169+t**2)]
            return rates
        rateMaxFour = np.array([2,2,2.23,2.31])
        minBoundsFour = np.array([0.4,0.2,0.1,0.1])
        numProcesses = [1,2,4]

        nH_timing_simulation(j, [updateFunctionOne,updateFunctionTwo,updateFunctionFour], [rateFunctionSingOne,rateFunctionSingTwo,rateFunctionSingFour], 
        [rateFunctionVectOne,rateFunctionVectTwo,rateFunctionVectFour], competing_inverse, nMGA, competing_thin, gMax, [rateMaxOne, rateMaxTwo, rateMaxFour],
        [minBoundsOne, minBoundsTwo, minBoundsFour], numProcesses, tMax)

if __name__=="__main__":
    main(False, True, False)
