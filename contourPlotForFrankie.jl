#=
A script given to Frankie Patten-Elliot that contained the code used to create
the heatmap/contour plot of model belief on seeing N number of cases again, after
a given date. 

Author: Joel Trent
=#

using Dates, PlotlyJS

function indexWhereAllValuesLessThanX(array, x)
    #=
    Given an array, return the largest range that represents the range array[index:end]
    where all values in this range are less than x

    If no range exists, return 0:0
    =#
    for i in length(array):-1:1
        if array[i] >= x
            if i+1 > length(array)
                return 0:0
            else
                return i+1:length(array)
            end
        end
    end

    return 1:length(array)
end

function testIndexWhereAll()
    @assert indexWhereAllValuesLessThanX([1,2,3,4,5], 6) == 1:5
    @assert indexWhereAllValuesLessThanX([5,4,3,2,1], 4) == 3:5
end

function probOfLessThanXGivenYDays(dailyConfirmedCases, x, y::Union{UnitRange,StepRange})
    #=
    Given a 2D array, where the columns contain individual realisations and the
    rows represent values at given times, return the proportion of columns that
    satisfy indexWhereAllValuesLessThanX after y days.

    The first row corresponds to t=0
    The second row corresponds to t=1 etc.
    =#

    probability = zeros(length(y))

    caseXranges = [0:0 for _ in 1:length(dailyConfirmedCases[1,:])]

    for col in 1:length(dailyConfirmedCases[1,:])
        caseXranges[col] = indexWhereAllValuesLessThanX(dailyConfirmedCases[:,col], x)
    end

    for i in 1:length(y)
        for j in caseXranges
            if !isnothing(j) && y[i] in j
                probability[i] += 1.0
            end
        end
        probability[i] = probability[i] / length(caseXranges)
    end

    return probability
end

function testprobOfLess()
    @assert sum(probOfLessThanXGivenYDays([0 1; 2 3; 5 6; 4 5; 3 2; 2 1; 1 0], 6, 1:7) .== [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0])==7
    @assert sum(probOfLessThanXGivenYDays([0 1; 2 3; 5 6; 4 5; 3 2; 2 1; 1 0], 5, 1:7) .== [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])==7
end

function caseNumbersBeginToDrop(dailyConfirmedCases, numCasesRange=0:1, daysSinceRange=0:1)

    # numCasesRange = 10:60
    # daysSinceRange = 45:70
    probabilities = zeros(length(numCasesRange), length(daysSinceRange))

    for i in 1:length(numCasesRange)
        probabilities[i,:] = probOfLessThanXGivenYDays(dailyConfirmedCases, numCasesRange[i], daysSinceRange)
    end

    return probabilities
end

function probOfLessThanXGivenYContour(probabilities, numCasesRange, daysSinceRange,
    title, outputFileName, Display=true, save=false, useDates=false)

    #=
    replace y=collect(numCasesRange) with the appropriate vector of y values
    e.g. y = np.array([0, 5, 10, 15]) etc.
    same with x = xValues
    collect function takes a Julia range and converts it to an array form,
    similar to linspace.
    =#

    xValues = []
    xTitle = ""
    if useDates
        # dayZeroDate = Date("17-08-2021", dateformat"d-m-y")
        xValues = [Dates.format(DAY__ZERO__DATE + Dates.Day(i), "u d") for i in collect(daysSinceRange)]
        xTitle = "Date"
    else
        xValues = collect(daysSinceRange).
        xTitle = "Days since detection"
    end

    fig = PlotlyJS.plot(PlotlyJS.contour(
        x=xValues, # horizontal axis
        y=collect(numCasesRange), # vertical axis
        z=probabilities,
        # heatmap gradient coloring is applied between each contour level
        contours=attr(
            coloring ="heatmap",
            showlabels = true, # show labels on contours
            labelfont = attr( # label font properties
                size = 12,
                color = "white",
            )
        ), colorbar=attr(
        nticks=10, ticks="outside",
        ticklen=5, tickwidth=2,
        showticklabels=true,
        tickangle=0, tickfont_size=15,title="Probability", # title here
        titleside="right",
        titlefont=attr(
            size=18,
            family="Arial, sans-serif"
        )
        )),
        Layout(width=5*150, height=4*150, font=attr(
            size=16,
            family="Arial, sans-serif"
        ), title=title,
        yaxis=attr(title_font=attr(size=16), ticks="outside", tickwidth=2, ticklen=5, col=1,showline=true, linewidth=2, linecolor="black", mirror=true),
        xaxis=attr(title_font=attr(size=16), nticks=7, ticks="outside", tickwidth=2, ticklen=5, col=1, showline=true, linewidth=2, linecolor="black", mirror=true),
        xaxis_title=xTitle, yaxis_title="Less than N cases per day")

    )


    # if Display
    #     # required to display graph on plots.
    #     display(fig)
    # end
    if save
        # Save graph as pngW
        PlotlyJS.savefig(fig, outputFileName*".png", width=5*150, height=4*150,scale=2)

    end
    close()
end

function main()

    Display=true
    save=true

    numCasesRange = 80:-10:10
    daysSinceRange = 10:5:40

    probabilities = caseNumbersBeginToDrop(dailyDetectedCases, numCasesRange, daysSinceRange)

    title = "Probability of less than N cases per day, x days after detection"
    outputFileName = "./ProbDaysSinceDetection_August2021"
    probOfLessThanXGivenYHeatMap(probabilities, numCasesRange, daysSinceRange, title, outputFileName, Display, save)

end

main()
