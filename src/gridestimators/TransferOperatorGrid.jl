export TransferOperatorGrid, _transferentropy
import StatsBase
import CausalityToolsBase: RectangularBinning, BinningHeuristic
import GroupSlices: groupslices, groupinds

"""
    TransferOperatorGrid(; b::Number = 2, summary_statistic = StatsBase.mean, 
        binning = ExtendedPalusLimit())

An transfer entropy estimator which computes transfer entropy over a 
dicretization of an appropriate delay reconstruction of the input data.
Invariant probabilities over the partition are computed using an 
approximation to the transfer (Perron-Frobenius) operator over the 
grid [^Diego2019], which explicitly gives the transition probabilities between states. 

## Fields 

- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).
- **`summary_statistic::Function = StatsBase.mean`**: The summary statistic to use if multiple discretization schemes are given.
- **`binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic} = ExtendedPalusLimit()`**: 
    The discretization scheme. Can either be fixed (i.e. one or more `RectangularBinning` instances),
    or a `BinningHeuristic`. In the latter case, the binning is determined from the input data.

[^Diego2019]: Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
Base.@kwdef struct TransferOperatorGrid <: BinningTransferEntropyEstimator
    """ The base of the logarithm usen when computing transfer entropy. """
    b::Number = 2.0

    """ The summary statistic to use if multiple discretization schemes are given """
    summary_statistic::Function = StatsBase.mean

    """ The discretization scheme. """
    binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic} = ExtendedPalusLimit()

    TransferOperatorGrid(b, summary_statistic, binning) = new(b, summary_statistic, binning)
end


import PerronFrobenius: InvariantDistribution, invariantmeasure, SingleGrid
import CausalityToolsBase: get_minima_and_edgelengths, encode 
import .._transferentropy
import ..TEVars


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
    
    n = size(marginal_inds, 1)
    marginal = zeros(Float64, n)
    @inbounds for i = 1:n
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
    μ = invariantmeasure(pts, SingleGrid(binning))
    
    # Find the unique visited bins, then find the subset of those bins 
    # with nonzero measure.
    
    # Multiple points may fall inside each bin, so encode all points
    # into an integer tuple indicating the bin they fall in.
    mini, edgelengths = get_minima_and_edgelengths(pts, binning)
    encoded_pts = encode(pts, mini, edgelengths)
    
    positive_measure_bins = unique(encoded_pts)[μ.nonzero_inds]

    p_Y  = marginal(Y, Array(transpose(hcat(positive_measure_bins...,))), μ)
    p_XY = marginal(XY, Array(transpose(hcat(positive_measure_bins...,))), μ)
    p_YZ = marginal(YZ, Array(transpose(hcat(positive_measure_bins...,))), μ)

    # Base of the logarithm
    b = estimator.b

    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(μ.dist[μ.nonzero_inds], b)
end
