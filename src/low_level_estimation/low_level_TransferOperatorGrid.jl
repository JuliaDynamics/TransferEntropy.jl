"""
	marginal_indices(A)

Returns a column vector with the same number of elements as there are unique
rows in A. The value of the ith element is the row indices of rows in A
matching the ith unique row.
"""
marginal_indices(A) = groupinds(groupslices(A, 1))

"""
    marginal(along_which_axes::Union{Vector{Int}, UnitRange{Int}},
            visited_bin_labels::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution)

Compute the marginal distributions along the specified axes, taking
into account the invariant distribution over the bins.
"""
function marginal(along_which_axes::Union{Vector{Int}, UnitRange{Int}},
                visited_bin_labels::Array{Int, 2},
                iv::InvariantDistribution)

    marginal_inds::Vector{Vector{Int}} =
        marginal_indices(visited_bin_labels[:, along_which_axes])

    # Only loop over the nonzero entries of the invariant distribution
    nonzero_elements_of_dist = iv.dist[iv.nonzero_inds]
    marginal = zeros(Float64, size(marginal_inds, 1))
    @inbounds for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(nonzero_elements_of_dist[marginal_inds[i]])
    end

    return marginal
end


function _transferentropy(pts, vars::TEVars, binning::RectangularBinning, estimator::TransferOperatorGrid)

    # Collect variables for the marginals 
    C = vars.T
    XY = [vars.𝒯; vars.T; C]
    YZ = [vars.T; vars.S; C]
    Y =  [vars.T; C]

    # Calculate the invariant distribution over the bins.
    μ = invariantmeasure(pts, binning)
    
    # Find the unique visited bins, then find the subset of those bins 
    # with nonzero measure.
    positive_measure_bins = unique(μ.encoded_points)[μ.measure.nonzero_inds]

    p_Y  = marginal(Y, Array(transpose(hcat(positive_measure_bins...,))), μ.measure)
    p_XY = marginal(XY, Array(transpose(hcat(positive_measure_bins...,))), μ.measure)
    p_YZ = marginal(YZ, Array(transpose(hcat(positive_measure_bins...,))), μ.measure)

    # Base of the logarithm
    b = estimator.b

    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(μ.measure.dist[μ.measure.nonzero_inds], b)
end
