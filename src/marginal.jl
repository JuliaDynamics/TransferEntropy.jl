"""
Returns a column vector with the same number of elements as there are unique
rows in A. The value of the ith element is the row indices of rows in A
matching the ith unique row.
"""
marginal_indices(A) = GroupSlices.groupinds(GroupSlices.groupslices(A, 1))

"""
How many times does each unique row in A appear? Returns a column vector with the same
number of elements as there are unique rows in A. The value of the ith element of the
return vector is the number of times the ith unique row of A appears in A.
"""
marginal_multiplicity(A) = [length(x) for x in marginal_indices(A)]

"""
Compute the marginal for a binning with an associated transfer operator. The
    marginal is computed for the columns `cols`.
"""
function marginal(cols::Vector{Int},
                    rb::StateSpaceReconstruction.RectangularBinning,
                    to::PerronFrobenius.RectangularBinningTransferOperator,
                    iv::PerronFrobenius.InvariantDistribution)
    positive_measure_bins = rb.unique_nonempty_bins[iv.nonzero_inds, :]

    # Loop over the positively measured bins.
    marginal_inds::Vector{Vector{Int}} = marginal_indices(positive_measure_bins[:, cols])
    marginal = zeros(Float64, size(marginal_inds, 1))

    for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(iv.dist[iv.nonzero_inds][marginal_inds[i]])
    end

    return marginal
end

function marginal(cols::Vector{Int},
                    positive_measure_bins::Array{Int, 2},
                    iv::PerronFrobenius.InvariantDistribution)

    marginal_inds::Vector{Vector{Int}} = marginal_indices(positive_measure_bins[:, cols])

    # Only loop over the nonzero entries of the invariant distribution
    nonzero_elements_of_dist = iv.dist[iv.nonzero_inds]
    marginal = zeros(Float64, size(marginal_inds, 1))
    @inbounds for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(nonzero_elements_of_dist[marginal_inds[i]])
    end
    return marginal
end

export marginal_indices, marginal_multiplicity, marginal
