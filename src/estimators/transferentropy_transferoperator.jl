"""
	marginal_indices(A)

Returns a column vector with the same number of elements as there are unique
rows in A. The value of the ith element is the row indices of rows in A
matching the ith unique row.
"""
marginal_indices(A) = GroupSlices.groupinds(GroupSlices.groupslices(A, 1))

"""
    marginal(along_which_axes::Union{Vector{Int}, UnitRange{Int}},
            visited_bin_labels::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution)

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
    transferentropy(bins_visited_by_orbit::Array{Int, 2},
                    iv::PerronFrobenius.InvariantDistribution,
                    vars::TEVars)

Using the invariant probability distribution obtained from the
transfer operator to the visited bins of the partitioned state
space. `bins_visited_by_orbit` are the bin labels assigned to
each element of the partition that gets visited by the orbit.

We calculate transfer entropy from the embedding
`E`, given a discretization scheme controlled by `\epsilon` and
information `v::TEVars` about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.

"""
function transferentropy_transferoperator_visitfreq(
            bins_visited_by_orbit::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution,
            v::TEVars)

    # Verify that the number of dynamical variables in
    # the embedding agrees with the number of dynamical
    # variables in `v`(which indicates which variables of
    # the embedding should be considered part of the
    # transfer entropy computation). If these do not
    # agree, the marginals are computed wrongly, and
    # the transfer entropy estimate is not correct
    # (it could even be a negative value, which is
    # nonsensical).
    nvars = length(v.target_future) +
            length(v.target_presentpast) +
            length(v.source_presentpast) +
            length(v.conditioned_presentpast)

    dim = size(bins_visited_by_orbit, 1) # one variable per row

    if dim != nvars
        warn(""" The embedding dimension does not agree with the
            provided $v. The transfer entropy estimate is not
            correct.
            """)
    end

    unique_visited_bins = transpose(unique(bins_visited_by_orbit, 2))
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
        ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
        v::TransferEntropy.TEVars) -> Float64

Using the transfer operator to calculate probability
distributions,  calculate transfer entropy from the embedding
`E`, given a discretization scheme controlled by `ϵ` and
information `v::TEVars` about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.
"""
function transferentropy_transferoperator_visitfreq(
                    E::AbstractEmbedding,
                    ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
                    v::TransferEntropy.TEVars)

    # Verify that the number of dynamical variables in
    # the embedding agrees with the number of dynamical
    # variables in `v`(which indicates which variables of
    # the embedding should be considered part of the
    # transfer entropy computation). If these do not
    # agree, the marginals are computed wrongly, and
    # the transfer entropy estimate is not correct
    # (it could even be a negative value, which is
    # nonsensical).
    nvars = length(v.target_future) +
    		length(v.target_presentpast) +
    		length(v.source_presentpast) +
    		length(v.conditioned_presentpast)

    dim = size(E.points, 1) # one variable per row

    if dim != nvars
        warn(""" The embedding dimension $dim does not agree with the
            provided $v. The transfer entropy estimate is not
            correct.
            """)
    end

    # Identify which bins of the partition resulting
    # from using ϵ each point of the embedding visits.
    bins_visited_by_orbit = assign_bin_labels(E, ϵ)

    # Which are the visited bins, which points
    # visits which bin, repetitions, etc...
    binvisits = organize_bin_labels(bins_visited_by_orbit)

    # Use that information to estimate transfer operator
    TO = PerronFrobenius.transferoperator(binvisits)
    invdist = left_eigenvector(TO)

    transferentropy_transferoperator_visitfreq(bins_visited_by_orbit, invdist, v)
end


"""
    transferentropy_transferoperator(E::AbstractEmbedding,
                ϵ::Vector{Union{Int, Float64, Vector{Float64}, Vector{Int}}},
                v::TEVars)

Compute transfer entropy over a range of bin sizes.
"""
function transferentropy_transferoperator_visitfreq(E::AbstractEmbedding,
        ϵ::Vector{Union{Int, Float64, Vector{Float64}, Vector{Int}}},
        v::TEVars)
    map(ϵᵢ -> transferentropy_transferoperator_visitfreq(E, ϵᵢ, v), ϵ)
end

# Shorter alias
tetofreq = transferentropy_transferoperator_visitfreq

#############################################################################
# One should be able to just provide some points that has been pre-embedded too
#############################################################################
"""
    transferentropy_transferoperator_visitfreq(pts::AbstractArray{T, 2},
        ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
        v::TransferEntropy.TEVars)

Using the transfer operator to calculate probability distributions,
calculate transfer entropy from the points `pts`,
given a discretization scheme controlled by `ϵ` and
information `v::TEVars` about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.
The points will be embedded behind the scenes.
"""
transferentropy_transferoperator_visitfreq(pts::AbstractArray{T, 2},
    ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
    v::TransferEntropy.TEVars) where T = tetofreq(embed(pts), ϵ, v)
