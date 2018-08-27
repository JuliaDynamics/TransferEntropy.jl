"""
    TransferEntropyVariables(XY, YZ, Y, W)


X = future of target variable(s), times (t + τ₁, t + τ₂, ...) where τᵢ ≧ 0.
Y = present and past of target variable(s), at times (t, t + η₁, t + η₂, ...) where ηᵢ ≦ 0.
Z = source variable(s) at times (t, t + ζ₁, t + ζ₂, ...) where ζᵢ ≦ 0.
W = conditioned variable(s) at times (t, t - π₁, t - π₂, ...) where πᵢ ≦ 0.
"""
struct TransferEntropyVariables
    XY::Vector{Int} #
    YZ::Vector{Int} #
    Y::Vector{Int} #
    W::Vector{Int} #  # extra variables
end


"""
transferentropy(b::StateSpaceReconstruction.RectangularBinning,
                invdist::PerronFrobenius.InvariantDistribution,
                vars::TransferEntropyVariables) -> Float64

The workhorse function for all transfer entropy estimators.

Computes transfer entropy from a binning and an associated invariant probability
distributionon the elements of the partition. This allows for using invariant distributions
obtained by means other than the transfer operator.
"""
function transferentropy(eqb::StateSpaceReconstruction.RectangularBinning,
            iv::PerronFrobenius.InvariantDistribution,
            vars::TransferEntropyVariables)

    positive_measure_bins = eqb.unique_nonempty_bins[iv.nonzero_inds, :]

    p_Y  = marginal([vars.Y;  vars.W], positive_measure_bins, iv)
    p_XY = marginal([vars.XY; vars.W], positive_measure_bins, iv)
    p_YZ = marginal([vars.YZ; vars.W], positive_measure_bins, iv)

    ((entropy(p_YZ) +
        entropy(p_XY) -
        entropy(p_Y)) -
        entropy(iv.dist[iv.nonzero_inds])) / log(2)
end

"""
Compute transfer entropy.
"""
function transferentropy(unique_nonempty_bins::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution,
            vars::TransferEntropyVariables)

    positive_measure_bins = unique_nonempty_bins[iv.nonzero_inds, :]
    p_Y  = marginal([vars.Y;  vars.W], positive_measure_bins, iv)
    p_XY = marginal([vars.XY; vars.W], positive_measure_bins, iv)
    p_YZ = marginal([vars.YZ; vars.W], positive_measure_bins, iv)

    ((entropy(p_YZ) +
        entropy(p_XY) -
        entropy(p_Y)) -
        entropy(iv.dist[iv.nonzero_inds])) / log(2)
end

"""
    transferentropy(E::StateSpaceReconstruction.Embedding,
                    n_subdivisions::Vector{Int},
                    vars::TransferEntropyVariables) -> Vector{Float64}

Estimate transfer entropy from an embedding over multiple binsizes. The bin sizes are
given as a vector of integers `n_subdivisions`, where each element indicates how many
equidistant intervals the axes of the embedding should be divided into.

This results in `n_subdivisions` different partitionings of the embedding. For each of
these partitionings, the transfer operator is estimated, and from the transfer operator
the associated invariant distribution on the elements of the partition is estimated.
Lastly, the transfer entropy is computed for each of the resulting invariant distributions.

## Returns
A vector of transfer entropy estimate, one for each partition.
"""
function transferentropy(E::StateSpaceReconstruction.Embedding,
                        n_subdivisions::Vector{Int},
                        vars::TransferEntropyVariables)

    n = length(n_subdivisions)
    binnings = Vector{StateSpaceReconstruction.RectangularBinning}(n)
    transferoperators = Vector{PerronFrobenius.AbstractTransferOperator}(n)
    invariantdists = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        binnings[i] = bin_rectangular(E, n_subdivisions[i])
        transferoperators[i] = transferoperator(binnings[i])
        invariantdists[i] = PerronFrobenius.left_eigenvector(transferoperators[i])
    end

    for i = 1:n
        transferentropies[i] = transferentropy(binnings[i], invariantdists[i], vars)
    end

    return transferentropies
end

"""
    transferentropy(binnings::Vector{StateSpaceReconstruction.RectangularBinning},
                    vars::TransferEntropyVariables) -> Vector{Float64}

Estimate transfer entropy from a set of precomputed binnings. Behind the scenes, this
function estimates the transfer operator and associated invariant measures over the
elements of each of the partitions, then estimates transfer entropy from each of those
probability distributions. Returns a vector of transfer entropy estimates, one for each
of the binnings.
"""
function transferentropy(binnings::Vector{StateSpaceReconstruction.RectangularBinning},
                        vars::TransferEntropyVariables)

    n = length(binnings)
    transferoperators = Vector{PerronFrobenius.TransferOperator}(n)
    invariantdists = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        transferoperators[i] = transferoperator(binnings[i])
        invariantdists[i] = PerronFrobenius.left_eigenvector(transferoperators[i])
    end

    for i = 1:n
        transferentropies[i] = transferentropy(binnings[i], invariantdists[i], vars)
    end

    return transferentropies
end

"""
    transferentropy(E::StateSpaceReconstruction.Embedding,
                    nbins_eachaxis::Int,
                    vars::TransferEntropyVariables) -> Float64

Compute transfer entropy from an embedding, given a number of bins `nbins_eachaxis` along
each axis of the embedding. Behind the scenes, this function overlays a rectangular grid
on the embedding, estimates the transfer operator, and from it obtains an invariant
probability distribution over the elements of the partition. These probabilities are
then used to estimate the transfer entropy.
"""
function transferentropy(E::StateSpaceReconstruction.Embedding,
                        nbins_eachaxis::Int,
                        vars::TransferEntropyVariables)
    b = bin_rectangular(E, nbins_eachaxis)
    to = PerronFrobenius.transferoperator(b)
    invariantdistribution = PerronFrobenius.left_eigenvector(to)
    transferentropy(b, invariantdistribution, vars)
end

"""
    transferentropy(b::StateSpaceReconstruction.RectangularBinning,
                        vars::TransferEntropyVariables) -> Float64

Compute transfer entropy from a rectangular binning.
"""
function transferentropy(b::StateSpaceReconstruction.RectangularBinning,
                        vars::TransferEntropyVariables)
    to = PerronFrobenius.transferoperator(b)
    invdist = PerronFrobenius.left_eigenvector(to)
    transferentropy(b, invdist, vars)
end

"""
    transferentropy(b::StateSpaceReconstruction.RectangularBinning,
                        to::PerronFrobenius.AbstractTransferOperator,
                        vars::TransferEntropyVariables) -> Float64

Compute transfer entropy from a binning and an associated transfer operator.
"""
function transferentropy(b::StateSpaceReconstruction.RectangularBinning,
                        to::PerronFrobenius.AbstractTransferOperator,
                        vars::TransferEntropyVariables)
    invdist = PerronFrobenius.left_eigenvector(to)
    transferentropy(b, invdist, vars)
end

######################################################
# UPDATED TO use TEVars, do this for the rest also
######################################################

function transferentropy(eqb::StateSpaceReconstruction.RectangularBinning,
            iv::PerronFrobenius.InvariantDistribution,
            vars::TEVars)
    v = vars

    C = vars.conditioned_presentpast
    XY = [v.target_future;      v.target_presentpast; C]
    YZ = [v.target_presentpast; v.source_presentpast; C]
    Y =  [v.target_presentpast;                       C]

    positive_measure_bins = eqb.unique_nonempty_bins[iv.nonzero_inds, :]

    p_Y  = marginal(Y, positive_measure_bins, iv)
    p_XY = marginal(XY, positive_measure_bins, iv)
    p_YZ = marginal(YZ, positive_measure_bins, iv)
    ((nat_entropy(p_YZ) +
        nat_entropy(p_XY) -
        nat_entropy(p_Y)) -
        nat_entropy(iv.dist[iv.nonzero_inds])) / log(2)
end

function transferentropy(E::StateSpaceReconstruction.AbstractEmbedding,
                            binsize::Union{Int, Float64, Vector{Float64}},
                            vars::TEVars)

    b = bin_rectangular(E, binsize)
    to = transferoperator(b)
    iv = left_eigenvector(to)
    transferentropy(b, iv, vars)
end
