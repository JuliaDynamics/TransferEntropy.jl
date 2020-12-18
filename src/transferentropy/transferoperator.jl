
import Entropies: TransferOperator, invariantmeasure, InvariantMeasure, Probabilities
using Entropies.GroupSlices
export TransferOperator

"""
	marginal_indices(x)

Returns a column vector `v` with the same number of elements as there are unique
elements in `x`. `v[i]` is the indices of elements in `x` matching `v[i]`. 

For example, if the third unique element in `x`, and the element `u₃ = unique(x)[3]` 
appears four times in `x`, then `v[3]` is a vector of four integers indicating the 
position of the elements matching `u₃`.
"""
function marginal_indices(visited_bins, selected_axes)
    marginal_pts = [x[selected_axes] for x in visited_bins]
    groupinds(groupslices(marginal_pts))
end

""" 
    marginal_probs_from_μ(seleced_axes, visited_bins, iv::InvariantMeasure, inds_μpositive)

Estimate marginal probabilities from a pre-computed invariant measure, given a set 
of visited bins, an invariant measure and the indices of the positive-measure bins.
The indices in `selected_axes` determines which marginals are selected.
"""
function marginal_probs_from_μ(seleced_axes, visited_bins, iv::InvariantMeasure, inds_μpositive)

    marginal_inds::Vector{Vector{Int}} = 
        marginal_indices(visited_bins, seleced_axes)

    # When the invariant measure over the joint space is already known, we don't
    # need to estimate histograms. We simply sum over the nonzero entries of the 
    # (already estimated) invariant distribution `iv` in the marginal space
    # (whose indices are given by `seleced_axes`).
    μpos = iv.ρ[inds_μpositive]
    marginal = zeros(Float64, length(marginal_inds))
    @inbounds for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(μpos[marginal_inds[i]])
    end
    return marginal
end

function transferentropy(s, t, est::TransferOperator{<:RectangularBinning}; base = 2, q = 1, 
        τT = -1, τS = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1)

    emb = EmbeddingTE(τT = τT, τS = τS, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯)

    joint_pts, vars, τs, js = te_embed(s, t, emb)
    iv = invariantmeasure(joint_pts, est.ϵ)

    # The bins visited by the orbit 
    unique_visited_bins = unique(iv.to.bins)

    # # The subset of visited bins with nonzero measure
    inds_μpositive = findall(iv.ρ .> 0)
    positive_measure_bins = unique_visited_bins[inds_μpositive]

    # Estimate marginal probability distributions from joint measure
    cols_ST = [vars.S; vars.T]
    cols_T𝒯 = [vars.𝒯; vars.T]
    cols_T = vars.T
    p_T  = marginal_probs_from_μ(cols_T, positive_measure_bins, iv, inds_μpositive)
    p_ST = marginal_probs_from_μ(cols_ST, positive_measure_bins, iv, inds_μpositive)
    p_T𝒯 = marginal_probs_from_μ(cols_T𝒯, positive_measure_bins, iv, inds_μpositive)
    p_joint = iv.ρ[inds_μpositive]

    te = genentropy(Probabilities(p_ST), base = base, q = q) + 
        genentropy(Probabilities(p_T𝒯), base = base, q = q) -
        genentropy(Probabilities(p_T), base = base, q = q) -
        genentropy(Probabilities(p_joint), base = base, q = q)
end


function transferentropy(s, t, c, est::TransferOperator{<:RectangularBinning}; base = 2, q = 1, 
        τT = -1, τS = -1, τC = -1, η𝒯 = 1, dT = 1, dS = 1, dC = 1, d𝒯 = 1)

    emb = EmbeddingTE(τT = τT, τS = τS, τC = τC, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯, dC = dC)

    joint_pts, vars, τs, js = te_embed(s, t, c, emb)
    iv = invariantmeasure(joint_pts, est.ϵ)

    # The bins visited by the orbit 
    unique_visited_bins = unique(iv.to.bins)

    # # The subset of visited bins with nonzero measure
    inds_μpositive = findall(iv.ρ .> 0)
    positive_measure_bins = unique_visited_bins[inds_μpositive]

    # Estimate marginal probability distributions from joint measure
    cols_ST = [vars.S; vars.T; vars.C]
    cols_T𝒯 = [vars.𝒯; vars.T; vars.C]
    cols_T = [vars.T; vars.C]
    p_T  = marginal_probs_from_μ(cols_T, positive_measure_bins, iv, inds_μpositive)
    p_ST = marginal_probs_from_μ(cols_ST, positive_measure_bins, iv, inds_μpositive)
    p_T𝒯 = marginal_probs_from_μ(cols_T𝒯, positive_measure_bins, iv, inds_μpositive)
    p_joint = iv.ρ[inds_μpositive]

    te = genentropy(Probabilities(p_ST), base = base, q = q) + 
        genentropy(Probabilities(p_T𝒯), base = base, q = q) -
        genentropy(Probabilities(p_T), base = base, q = q) -
        genentropy(Probabilities(p_joint), base = base, q = q)
end

