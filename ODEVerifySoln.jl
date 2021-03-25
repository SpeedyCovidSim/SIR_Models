#=
This is a code base for a the ODE solution to a well-mixed SIR simulation

Author: Joel Trent
=#

#Pkg.add("DifferentialEquations")
#Pkg.add("Interpolations")

using DifferentialEquations, Plots, Dierckx, StatsBase #Interpolations

push!( LOAD_PATH, "./" )
using sirModels: gillespieDirect2Processes_dist


# solve the SIR system of equations
function ODEsir!(du,u,p,t)
    # let S = u[1], I = u[2], R = u[3]
    alpha = 0.15
    beta = 1.5 / 10000
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

#sol([1,2,3])

#plot(sol)

Solution = sol(times)
ODESolution = Solution.u

println("Beginning simulation")


numSims = 1000
# Julia is column major
StStep = zeros(Int(tspan[end]/tStep+1),numSims, )
ItStep = copy(StStep)
RtStep = copy(StStep)
i = 1
for i = 1:numSims

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

println("Finished Simulation")

Smean = mean(StStep, dims = 2)
Imean = mean(ItStep, dims = 2)
Rmean = mean(RtStep, dims = 2)



ODEarray = vectorToArray(ODESolution)

misfitS, misfitI, misfitR = meanAbsError(Smean, Imean, Rmean, ODEarray)
println("Mean Abs Error S = $misfitS, Mean Abs Error I = $misfitI, Mean Abs Error R = $misfitR, ")
