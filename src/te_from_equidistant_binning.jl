"""
    TransferEntropyVariables(XY, YZ, Y, W)

"""
struct TransferEntropyVariables
    XY::AbstractArray{Int, 1} #
    YZ::AbstractArray{Int, 1} #
    Y::AbstractArray{Int, 1}  #
    W::AbstractArray{Int, 1}  # extra variables
end

"""
Returns a column vector with the same number of elements as there are unique
rows in V. The value of the ith element is the row indices of rows in V
matching the ith unique row.
"""
marginal_indices(V) = GroupSlices.groupinds(GroupSlices.groupslices(V, 1))

"""
How many times does each unique row in V appear? Returns a column vector with the same
number of elements as there are unique rows in V. The value of the ith element of the
return vector is the number of times the ith unique row of V appears in V.
"""
marginal_multiplicity(V) = [length(x) for x in marginal_indices(V)]

"""
Compute entropy of a probability distribution.
"""
function nat_entropy(prob::Vector{T}) where T<:Number
    te = 0.0

    @inbounds for i = 1:size(prob, 1)
        te -= prob[i] * log(prob[i])
    end

    return te
end

"""
Compute the marginal for a binning with an associated transfer operator. The
    marginal is computed for the columns `cols`.
"""
function marginal(cols::Vector{Int},
                    eqb::StateSpaceReconstruction.RectangularBinning,
                    to::PerronFrobenius.RectangularBinningTransferOperator,
                    iv::PerronFrobenius.InvariantDistribution)

    # Loop over the positively measured bins.
    marginal_inds = marginal_indices(eqb.positive_measure_bins[iv.nonzero_inds, cols])
    marginal = zeros(Float64, size(marginal_inds, 1))

    for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(iv.dist[iv.nonzero_inds][marginal_inds[i]])
    end

    return marginal
end

function marginal(cols::Vector{Int},
                    positive_measure_bins::Array{Int, 2},
                    iv::PerronFrobenius.InvariantDistribution)

    marginal_inds = marginal_indices(positive_measure_bins[:, cols])
    # Find the nonzero elements of the invariant distribution, loop
    # only over those.
    nonzero_elements_of_dist = iv.dist[iv.nonzero_inds]
    marginal = zeros(Float64, size(marginal_inds, 1))
    @inbounds for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(nonzero_elements_of_dist[marginal_inds[i]])
    end
    return marginal
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

    ((nat_entropy(p_YZ) +
        nat_entropy(p_XY) -
        nat_entropy(p_Y)) -
        nat_entropy(iv.dist[iv.nonzero_inds])) / log(2)
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

    ((nat_entropy(p_YZ) +
        nat_entropy(p_XY) -
        nat_entropy(p_Y)) -
        nat_entropy(iv.dist[iv.nonzero_inds])) / log(2)
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

    n = length(binsizes)
    binnings = Vector{StateSpaceReconstruction.RectangularBinning}(n)
    transferoperators = Vector{PerronFrobenius.TransferOperator}(n)
    invariantdists = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        binnings[i] = bin_rectangular(E, binsizes[i])
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


"""
Compute the transfer entropy resulting only from the geometry of the reconstructed
attractor. How? Assign uniformly distributed states on the volumes of the
reconstructed state space with nonzero measure.
"""
function shape_transferentropy(bins::Array{Int, 2}, vars::TransferEntropyVariables)
   dim, n_nonempty_bins = size(bins, 2), size(bins, 1)

   n_XY = marginal_multiplicity(bins[:, [vars.XY; vars.W]])
   n_Y  = marginal_multiplicity(bins[:, [vars.Y;  vars.W]])
   n_YZ = marginal_multiplicity(bins[:, [vars.YZ; vars.W]])

   # Transfer entropy as the sum of the marginal entropies
   ((nat_entropy(n_YZ) + nat_entropy(n_XY) - nat_entropy(n_Y)) / n_nonempty_bins) / log(2)
end

function shape_transferentropy(eqb::StateSpaceReconstruction.RectangularBinning,
                               vars::TransferEntropyVariables)

    shape_transferentropy(eqb.unique_nonempty_bins, vars)
end
