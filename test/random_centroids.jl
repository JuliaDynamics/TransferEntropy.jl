using Distributions
using InvariantDistribution
using SimplexSplitting
using TransferEntropy
using PlotlyJS


function te_over_binsizes(binsizes; npts::Int = 1000, covariance = 0.4)
    # Create an uncorrelated source and target
    dist = Normal()

    source = rand(dist, npts, 1)
    dest = covariance .* source[1:end] .+ (1.0 - covariance) .* rand(dist, npts, 1)

    # Embedding
    e = hcat(source[1:end-1], source[2:end], dest[2:end])

    if invariantset(e)
      #println("The embedding forms an invariant set. Continuing.")
    else
      #println("The embedding does not form an invariant set. Quitting.")
      return zeros(length(binsizes))
    end

    t = SimplexSplitting.triang_from_embedding(Embedding(e))
    SimplexSplitting.refine_variable_k!(t, mean(t.radii))
    M = mm_sparse(t)
    invmeasure, inds_nonzero_simplices = invariantdist(M)
    TE = zeros(length(binsizes))

    count = 0
    for binsize in binsizes
        count +=1
        te = te_from_triangulation(e, invmeasure, binsize)
        TE[count] = te
    end
    return TE
end

##################################################################
# Calculate TE for correlated gaussians over a range of bin sizes
##################################################################
binsizes = 1:1:200
covars = 0.3:0.2:0.5
reps = 3
npts = 120

TEs = zeros(Float64, length(binsizes), reps, length(covars))

for j = 1:length(covars)
    covar = covars[j]
    println("Covariance: ", covar, " Expected TE: ", log(1/(1 - covar^2)))
    for i = 1:reps
        println("\tRep #", i)
        te = te_over_binsizes(binsizes; npts = npts, covariance = covar)
        TEs[1:end, i, j] = te
    end
end

TEs

####################
# PLOT THE TE CURVES
####################
data = PlotlyJS.GenericTrace[]
layout = PlotlyJS.Layout(xaxis = PlotlyJS.attr(title = "Bin size", ticks = "outside"),
                          yaxis = PlotlyJS.attr(title = "TE (nats)"))
for j = 1:length(covars)
    push!(data, PlotlyJS.scatter(; x = binsizes, y = fill(log(1/(1-covars[j]^2)), length(binsizes)), name = string(covars[j])))
    for i = 1:reps
        trace = PlotlyJS.scatter(;x = binsizes,
                                   y = TEs[1:end, i, j],
                                   mode = "markers",
                                   marker = PlotlyJS.attr(size = covars[j]+5, color = "black"),
                                   name = string(covars[j]))
        push!(data, trace)
    end
end

TEs

PlotlyJS.plot(data, layout)
