using CSV, DataFrames
# using Pandas
# a = DataFrame(i=>[] for i in 1:20000)

a = DataFrame()
# a.two = 1

for i in 1:70000
    a[!,"$i"] = [i]
end

CSV.write("./test.csv", a, bufsize=2^25)

# aTranspose = DataFrame([[names(a)]; collect.(eachrow(a))], [:column; Symbol.(axes(a, 1))])
aTranspose = Df_transpose(a)

CSV.write("./testTranspose.csv", aTranspose, header=false, bufsize=2^25)
# Pandas.DataFrame(a)

function Df_transpose(df)
    #=
    Given a DataFrame, return it's transpose
    =#
    df = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])
    return select(df, 2:ncol(df))
end

df = a
df = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])

df = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])

select(df, 2:ncol(df))
