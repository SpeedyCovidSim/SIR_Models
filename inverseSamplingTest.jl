using Distributions
using Random


dist = Weibull(2.83, 5.67)

# 1.0
ccdf(dist, 0)

ccdf(dist, 5)


n = 1000
times = [0,1,2,3,4,5,6,7,8,9]
timeToReact = 0

for time in times
    # contribution to prob, of time already passed
    cdf_at_time = cdf(dist, time) # this is what we scale our rand() value by

    print(cdf_at_time)
    print("\n")

    rand_values = rand(n)

    transformed_values = rand_values .* (1-cdf_at_time) .+ cdf_at_time

    print(mean(transformed_values))
    print("\n")

    timeToReact = invlogcdf.(dist, log.(transformed_values)) .- time

    print(mean(timeToReact))
    print("\n\n")
end


log(0.9999)

invlogcdf(dist, log(transformed_value))

cdf_at_time = cdf(dist,0)
