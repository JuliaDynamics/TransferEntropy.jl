import PerronFrobenius:
    invariantmeasure,
    InvariantDistribution

import StateSpaceReconstruction: 
    groupinds, 
    groupslices

import StatsBase

"""
    marginal(along_which_axes::Union{Vector{Int}, UnitRange{Int}},
            visited_bin_labels::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution)

Compute the marginal distributions along the specified axes, taking
into account the invariant distribution over the bins.
"""
function marginal_respecting_μ(along_which_axes::Union{Vector{Int}, UnitRange{Int}},
                visited_bin_labels::Array{Int, 2},
                iv::InvariantDistribution)

    marginal_inds = visited_bin_labels[along_which_axes, :]
    grouped_indices = groupinds(groupslices(marginal_inds, 2))

    # Only loop over the nonzero entries of the invariant distribution
    nonzero_elements_of_dist = iv.dist[iv.nonzero_inds]
    marginal = zeros(Float64, size(marginal_inds, 2))
    @inbounds for i = 1:size(marginal_inds, 2)
        marginal[i] = sum(nonzero_elements_of_dist[marginal_inds[i]])
    end
    return marginal
end

"""
transferentropy(pts, ϵ, vars::TEVars, estimator::TransferOperatorGrid; b = 2)

Compute transfer entropy for a set of ordered points representing
an appropriate embedding of some time series. See documentation for 
`TEVars` for info on how to specify the marginals (i.e. which variables 
of the embedding are treated as what). 

`b` sets the base of the logarithm (e.g `b = 2` gives the transfer 
entropy in bits). 
"""
function transferentropy(pts::Vector{T}, ϵ, vars::TEVars, estimator::TransferOperatorGrid; 
        b = 2) where {T <: Vector, SVector, MVector}


    # Calculate the invariant distribution over the bins.
    μ = invariantmeasure(pts, ϵ)

    # Find the unique visited bins, then find the subset of those bins 
    # with nonzero measure.
    # Make a version of marginal that takes columns 
    positive_measure_bins = unique(μ.encoded_points)[μ.measure.nonzero_inds]

        # Collect variables for the marginals 
    C = vars.conditioned_presentpast
    XY = [vars.target_future;      vars.target_presentpast; C]
    YZ = [vars.target_presentpast; vars.source_presentpast; C]
    Y =  [vars.target_presentpast;                          C]

    p_Y  = marginal_respecting_μ(Y, positive_measure_bins, μ.measure)
    p_XY = marginal_respecting_μ(XY, positive_measure_bins, μ.measure)
    p_YZ = marginal_respecting_μ(YZ, positive_measure_bins, μ.measure)

    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(μ.measure.dist[μ.measure.nonzero_inds], b)
end