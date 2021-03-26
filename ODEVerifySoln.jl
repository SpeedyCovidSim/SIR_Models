#=
This is a code base for comparing the ODE solution to a well-mixed SIR simulation
to a randomly generated SIR solution, using the gillespie direct approach

Author: Joel Trent
=#

#Pkg.add("DifferentialEquations")
#Pkg.add("Interpolations")

using DifferentialEquations, Plots, Dierckx, StatsBase, LightGraphs, MetaGraphs #Interpolations

push!( LOAD_PATH, "./" )
using sirModels: gillespieDirect2Processes_dist, gillespieDirect_network!
using networkFunctions: initialiseNetwork!

# solve the SIR system of equations
function ODEsir!(du,u,p,t)
    # let S = u[1], I = u[2], R = u[3]
    alpha = 0.15
    beta = 1.5 / 10000
    du[1] = -beta*u[1]*u[2]
    du[2] = beta*u[1]*u[2]-alpha*u[2]
    du[3] = alpha*u[2]
end

function ODEnetwork!(du,u,p,t)
    # let S = u[1], I = u[2], R = u[3]
    alpha = 0.15
    beta = 1.5 / 1000
    du[1] = -beta*u[1]*u[2]
    du[2] = beta*u[1]*u[2]-alpha*u[2]
    du[3] = alpha*u[2]
end

function meanAbsError(Smean, Imean, Rmean, ODEarray)
    misfitS = sum(abs.(Smean - ODEarray[:,1]))/length(Smean)
    misfitI = sum(abs.(Imean - ODEarray[:,2]))/length(Imean)
    misfitR = sum(abs.(Rmean - ODEarray[:,3]))/length(Rmean)

    return misfitS, misfitI, misfitR
end

function vectorToArray(ODESolution)
    ODEarray = ODESolution[1]
    for i in 1:length(ODESolution)-1

        ODEarray = hcat(ODEarray, ODESolution[i+1])
    end

    return reverse(rotr90(ODEarray),dims=2)
end

function main(wellMixed, Network)

    println("Solving ODE")
    N = 10000

    Alpha = 0.15
    beta = 1.5 / N

    # initial SIR totals
    u0 = [N*0.95,N*0.05,0]

    # time span to solve on
    tspan = (0.0,40.0)
    tStep = 0.01

    # times to solve on
    times = [i for i=tspan[1]:tStep:tspan[end]]

    # solve the ODE
    prob = ODEProblem(ODEsir!, u0, tspan)
    sol = solve(prob)

    println("ODE Solved")

    #plot(sol)

    Solution = sol(times)
    ODESolution = Solution.u

    ODEarray = vectorToArray(ODESolution)

    if wellMixed

        println("Beginning simulation of well mixed")

        numSims = 2000
        # Julia is column major
        StStep = zeros(Int(tspan[end]/tStep+1),numSims)
        ItStep = copy(StStep)
        RtStep = copy(StStep)
        i = 1
        time = @elapsed Threads.@threads for i = 1:numSims

            # Verifying the well mixed solution
            t, S, I, R = gillespieDirect2Processes_dist(tspan[end], u0[1], u0[2], u0[3], Alpha, beta, N)

            # interpolate using linear splines
            splineS = Spline1D(t, S, k=1)
            splineI = Spline1D(t, I, k=1)
            splineR = Spline1D(t, R, k=1)

            StStep[:,i] = splineS(times)
            ItStep[:,i] = splineI(times)
            RtStep[:,i] = splineR(times)
        end

        println("Finished Simulation in $time seconds")

        Smean = mean(StStep, dims = 2)
        Imean = mean(ItStep, dims = 2)
        Rmean = mean(RtStep, dims = 2)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, ODEarray)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")
    end

    println("Solving ODE")
    N = 1000

    Alpha = 0.15
    beta = 1.5 / N

    # initial SIR totals
    u0 = [N*0.95,N*0.05,0]

    # time span to solve on
    tspan = (0.0,40.0)
    tStep = 0.01

    # times to solve on
    times = [i for i=tspan[1]:tStep:tspan[end]]

    # solve the ODE
    prob = ODEProblem(ODEnetwork!, u0, tspan)
    sol = solve(prob)

    println("ODE Solved")

    Solution = sol(times)
    ODESolution = Solution.u

    ODEarray = vectorToArray(ODESolution)

    if Network
        t_max = tspan[end]
        infectionProp = 0.05
        simType = "SIR_direct"
        gamma = 0

        # initialise the network
        network = MetaGraph(complete_graph(N))
        println("Network returned")

        println("Beginning simulation of network")

        numSims = 1000
        # Julia is column major
        StStep = zeros(Int(tspan[end]/tStep+1),numSims)
        ItStep = copy(StStep)
        RtStep = copy(StStep)

        for i = 1:numSims

            # Verifying the well mixed network solution

            networkVertex_dict, network_dict, stateTotals = initialiseNetwork!(network, infectionProp, simType, Alpha, beta, gamma)

            t, state_Totals = gillespieDirect_network!(t_max, network, Alpha, beta, N, networkVertex_dict, network_dict, stateTotals)

            # interpolate using linear splines
            splineS = Spline1D(t, state_Totals[1,:], k=1)
            splineI = Spline1D(t, state_Totals[2,:], k=1)
            splineR = Spline1D(t, state_Totals[3,:], k=1)

            StStep[:,i] = splineS(times)
            ItStep[:,i] = splineI(times)
            RtStep[:,i] = splineR(times)
        end

        println("Finished Simulation")

        Smean = mean(StStep, dims = 2)
        Imean = mean(ItStep, dims = 2)
        Rmean = mean(RtStep, dims = 2)

        misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, ODEarray)
        println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")

    end

end

main(true, true)
