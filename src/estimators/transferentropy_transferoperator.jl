"""
Returns a column vector with the same number of elements as there are unique
rows in A. The value of the ith element is the row indices of rows in A
matching the ith unique row.
"""
marginal_indices(A) = GroupSlices.groupinds(GroupSlices.groupslices(A, 1))

"""
Compute the marginal distributions along the specified axes, taking
into account the invariant distribution over the bins.
"""
function marginal(along_which_axes::Union{Vector{Int}, UnitRange{Int}},
                visited_bin_labels::Array{Int, 2},
                iv::PerronFrobenius.InvariantDistribution)

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


"""
    transferentropy(visited_bin_inds::Array{Int, 2},
                    iv::PerronFrobenius.InvariantDistribution,
                    vars::TEVars)

Using the invariant probability distribution obtained from the
transfer operator to the visited bins of the partitioned state
space.

We calculate transfer entropy from the embedding
`E`, given a discretization scheme controlled by `\epsilon` and
information `v::TEVars` about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.

"""
function transferentropy_transferoperator(
            visited_bin_inds::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution,
            v::TEVars)
    unique_visited_bins = transpose(unique(visited_bin_inds, 2))
    positive_measure_bins = unique_visited_bins[iv.nonzero_inds, :]

    C = v.conditioned_presentpast
    XY = [v.target_future;      v.target_presentpast; C]
    YZ = [v.target_presentpast; v.source_presentpast; C]
    Y =  [v.target_presentpast;                       C]

    p_Y  = marginal([Y;  C], positive_measure_bins, iv)
    p_XY = marginal([XY; C], positive_measure_bins, iv)
    p_YZ = marginal([YZ; C], positive_measure_bins, iv)

    ((entropy(p_YZ) +
        entropy(p_XY) -
        entropy(p_Y)) -
        entropy(iv.dist[iv.nonzero_inds])) / log(2)
end


"""
    transferentropy_transferoperator(
        E::AbstractEmbedding,
        ϵ::Union{Int, Float64, Vector{Float64}},
        v::TransferEntropy.TEVars) -> Float64

Using the transfer operator to calculate probability
distributions,  calculate transfer entropy from the embedding
`E`, given a discretization scheme controlled by `\epsilon` and
information `v::TEVars`about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.
"""

function transferentropy_transferoperator(
                    E::AbstractEmbedding,
                    ϵ::Union{Int, Float64, Vector{Float64}},
                    v::TransferEntropy.TEVars)


    # Identify which bins of the partition resulting
    # from using ϵ each point of the embedding visits.
    visited_bins = assign_bin_labels(E, ϵ)

    # Which are the visited bins, which points
    # visits which bin, repetitions, etc...
    binvisits = organize_bin_labels(visited_bins)

    # Use that information to estimate transfer operator
    TO = PerronFrobenius.transferoperator(binvisits)
    invdist = left_eigenvector(TO)

    transferentropy_transferoperator(visited_bins, invdist, v)
end
