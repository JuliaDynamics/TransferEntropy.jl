"""
    TransferEntropyVariables(XY, YZ, Y, W)

"""
struct TransferEntropyVariables
    XY::AbstractArray{Int, 1}
    YZ::AbstractArray{Int, 1}
    Y::AbstractArray{Int, 1}
    W::AbstractArray{Int, 1}
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
function nat_entropy(prob)
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
                    eqb::StateSpaceReconstruction.EquidistantBinning,
                    to::PerronFrobenius.EquidistantBinningTransferOperator,
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

    # Loop over the positively measured bins.
    marginal_inds = marginal_indices(positive_measure_bins[:, cols])
    marginal = zeros(Float64, size(marginal_inds, 1))

    for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(iv.dist[iv.nonzero_inds][marginal_inds[i]])
    end

    return marginal
end

"""
Compute transfer entropy.
"""
function transferentropy(eqb::StateSpaceReconstruction.EquidistantBinning,
            iv::PerronFrobenius.InvariantDistribution,
            inds::TransferEntropyVariables)

    positive_measure_bins = eqb.unique_nonempty_bins[iv.nonzero_inds, :]

    p_Y  = marginal([inds.Y;  inds.W], positive_measure_bins, iv)
    p_XY = marginal([inds.XY; inds.W], positive_measure_bins, iv)
    p_YZ = marginal([inds.YZ; inds.W], positive_measure_bins, iv)

    ((nat_entropy(p_YZ) +
        nat_entropy(p_XY) -
        nat_entropy(p_Y)) -
        nat_entropy(iv.dist[iv.nonzero_inds])) / log(2)
end

"""
Compute transfer entropy.
"""
function transferentropy(eqb::StateSpaceReconstruction.EquidistantBinning,
            iv::PerronFrobenius.InvariantDistribution,
            inds::TransferEntropyVariables)

    positive_measure_bins = eqb.unique_nonempty_bins[iv.nonzero_inds, :]
    p_Y  = marginal([inds.Y;  inds.W], positive_measure_bins, iv)
    p_XY = marginal([inds.XY; inds.W], positive_measure_bins, iv)
    p_YZ = marginal([inds.YZ; inds.W], positive_measure_bins, iv)

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
            inds::TransferEntropyVariables)

    positive_measure_bins = unique_nonempty_bins[iv.nonzero_inds, :]
    p_Y  = marginal([inds.Y;  inds.W], positive_measure_bins, iv)
    p_XY = marginal([inds.XY; inds.W], positive_measure_bins, iv)
    p_YZ = marginal([inds.YZ; inds.W], positive_measure_bins, iv)

    ((nat_entropy(p_YZ) +
        nat_entropy(p_XY) -
        nat_entropy(p_Y)) -
        nat_entropy(iv.dist[iv.nonzero_inds])) / log(2)
end

"""
Estimate transfer entropy from an embedding.
"""
function transferentropy(E::StateSpaceReconstruction.GenericEmbedding,
    binsizes::Vector{Int},
    inds::TransferEntropyVariables)

    n = length(binsizes)
    binnings = Vector{StateSpaceReconstruction.EquidistantBinning}(n)
    transferoperators = Vector{PerronFrobenius.TransferOperator}(n)
    invariantdists = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        binnings[i] = bin_equidistant(E, binsizes[i])
        transferoperators[i] = transferoperator(binnings[i])
        invariantdists[i] = PerronFrobenius.left_eigenvector(transferoperators[i])
    end

    for i = 1:n
        transferentropies[i] = transferentropy(binnings[i], invariantdists[i], inds)
    end

    return transferentropies
end

"""
Estimate transfer entropy from a set of precomputed binnings.
"""
function transferentropy(binnings::Vector{StateSpaceReconstruction.EquidistantBinning},
                        inds::TransferEntropyVariables)

    n = length(binnings)
    transferoperators = Vector{PerronFrobenius.TransferOperator}(n)
    invariantdists = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        transferoperators[i] = transferoperator(binnings[i])
        invariantdists[i] = PerronFrobenius.left_eigenvector(transferoperators[i])
    end

    for i = 1:n
        transferentropies[i] = transferentropy(binnings[i], invariantdists[i], inds)
    end

    return transferentropies
end

"""
Compute transfer entropy from an embedding.
"""
function transferentropy(E::StateSpaceReconstruction.GenericEmbedding,
                        binsize::Int,
                        inds::TransferEntropyVariables)
    b = bin_equidistant(E, binsize)
    transferoperator = PerronFrobenius.transferoperator(b)
    invariantdistribution = PerronFrobenius.left_eigenvector(transferoperator)
    transferentropy(b, invariantdistribution, inds)
end

"""
Compute transfer entropy from a binning.
"""
function transferentropy(b::StateSpaceReconstruction.EquidistantBinning,
                        inds::TransferEntropyVariables)
    transferoperator = PerronFrobenius.transferoperator(b)
    invariantdistribution = PerronFrobenius.left_eigenvector(transferoperator)
    transferentropy(b, invariantdistribution, inds)
end



"""
Compute the transfer entropy resulting only from the geometry of the reconstructed
attractor. How? Assign uniformly distributed states on the volumes of the
reconstructed state space with nonzero measure.
"""
function shape_transferentropy(bins::Array{Int, 2}, inds::TransferEntropyVariables)
   dim, n_nonempty_bins = size(bins, 2), size(bins, 1)

   n_XY = marginal_multiplicity(bins[:, [inds.XY; inds.W]])
   n_Y  = marginal_multiplicity(bins[:, [inds.Y;  inds.W]])
   n_YZ = marginal_multiplicity(bins[:, [inds.YZ; inds.W]])

   # Transfer entropy as the sum of the marginal entropies
   ((nat_entropy(n_YZ) + nat_entropy(n_XY) - nat_entropy(n_Y)) / n_nonempty_bins) / log(2)
end

function shape_transferentropy(eqb::StateSpaceReconstruction.EquidistantBinning,
                               inds::TransferEntropyVariables)

    shape_transferentropy(eqb.unique_nonempty_bins, inds)
end
