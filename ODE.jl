#=
This is a code base for a the ODE solution to a well-mixed SIR simulation

Author: Joel Trent
=#

Pkg.add("DifferentialEquations")

using DifferentialEquations, Plots


# solve the SIR system of equations

function ODEsir!(du,u,p,t)
    # let S = u[1], I = u[2], R = u[3]
    alpha = 0.4
    beta = 10 ./ 10000
    du[1] = -beta*u[1]*u[2]
    du[2] = beta*u[1]*u[2]-alpha*u[2]
    du[3] = alpha*u[2]
end


u0 = [10000*0.95,10000*0.05,0]

tspan = (0.0,20.0)

prob = ODEProblem(ODEsir!, u0, tspan)

sol = solve(prob)

plot(sol)
