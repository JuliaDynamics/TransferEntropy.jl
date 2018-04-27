

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
                    to::PerronFrobenius.EquidistantBinningTransferOperator,
                    iv::PerronFrobenius.InvariantDistribution)

    # Loop over the positively measured bins.
    marginal_inds = marginal_indices(positive_measure_bins[:, cols])
    marginal = zeros(Float64, size(marginal_inds, 1))

    for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(iv.dist[iv.nonzero_inds][marginal_inds[i]])
    end

    return marginal
end



function te(eqb::StateSpaceReconstruction.EquidistantBinning,
            to::PerronFrobenius.EquidistantBinningTransferOperator,
            iv::PerronFrobenius.InvariantDistribution)

    positive_measure_bins = eqb.positive_measure_bins[iv.nonzero_inds, :]
    p_Y = marginal([2], positive_measure_bins, to, iv)
    p_XY = marginal([1, 2], positive_measure_bins, to, iv)
    p_YZ = marginal([2, 3], positive_measure_bins, to, iv)

    ((nat_entropy(p_YZ) +
        nat_entropy(p_XY) -
        nat_entropy(p_Y)) -
        nat_entropy(iv.dist[iv.nonzero_inds])) / log(2)


end

""" Estimate transfer entropy from an embedding. """
function te(E::StateSpaceReconstruction.GenericEmbedding, binsizes::Vector{Int})
    n = length(binsizes)
    binnings = Vector{StateSpaceReconstruction.EquidistantBinning}(n)
    transferoperators = Vector{PerronFrobenius.TransferOperator}(n)
    invariantdistributions = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        binnings[i] = bin_equidistant(E, binsizes[i])
        transferoperators[i] = transferoperator(binnings[i])
        invariantdistributions[i] = PerronFrobenius.left_eigenvector(transferoperators[i])
    end

    for i = 1:n
        transferentropies[i] = te(binnings[i], transferoperators[i], invariantdistributions[i])
    end

    return transferentropies
end

""" Estimate transfer entropy from a set of precomputed binnings. """
function te(binnings::Vector{StateSpaceReconstruction.EquidistantBinning})
    n = length(E)
    transferoperators = Vector{PerronFrobenius.TransferOperator}(n)
    invariantdistributions = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        transferoperators[i] = transferoperator(binnings[i])
        invariantdistributions[i] = PerronFrobenius.left_eigenvector(transferoperators[i])
    end

    for i = 1:n
        transferentropies[i] = te(binnings[i], transferoperators[i], invariantdistributions[i])
    end

    return transferentropies
end

""" Compute transfer entropy from an embedding. """
function te(E::StateSpaceReconstruction.GenericEmbedding, binsize::Int)
    b = bin_equidistant(E, binsize)
    transferoperator = PerronFrobenius.transferoperator(b)
    invariantdistribution = PerronFrobenius.left_eigenvector(transferoperator)
    te(b, transferoperator, invariantdistribution)
end

""" Compute transfer entropy from a binning. """
function te(b::StateSpaceReconstruction.EquidistantBinning)
    transferoperator = PerronFrobenius.transferoperator(b)
    invariantdistribution = PerronFrobenius.left_eigenvector(transferoperator)
    te(b, transferoperator, invariantdistribution)
end


"""
Compute the transfer entropy resulting only from the geometry of the reconstructed
attractor. How? Assign uniformly distributed states on the volumes of the
reconstructed state space with nonzero measure.
"""
function shape_te(bins::Array{Int, 2})
   dim, n_nonempty_bins = size(bins, 2), size(bins, 1)

   n_XY = marginal_multiplicity(bins[:, 1:2])
   n_Y  = marginal_multiplicity(bins[:, 2])
   n_YZ = marginal_multiplicity(bins[:, 2:3])

   # Transfer entropy as the sum of the marginal entropies
   ((nat_entropy(n_YZ) + nat_entropy(n_XY) - nat_entropy(n_Y)) / n_nonempty_bins) / log(2)
end

shape_te(eqb::StateSpaceReconstruction.EquidistantBinning) = shape_te(eqb.positive_measure_bins)
