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
    transferentropy_transferoperator_grid(
            bins_visited_by_orbit::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution,
            v::TEVars, normalise_to_tPP = false)

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

If `normalise_to_tPP = true`, then the TE estimate is normalised to
the entropy rate of the target variable, `H(target_future | target_presentpast)`.

"""
function transferentropy_transferoperator_grid(
            bins_visited_by_orbit::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution,
            v::TEVars; normalise_to_tPP = false)

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

    unique_visited_bins = transpose(unique(bins_visited_by_orbit, dims = 2))
    positive_measure_bins = unique_visited_bins[iv.nonzero_inds, :]

    C = v.conditioned_presentpast
    XY = [v.target_future;      v.target_presentpast; C]
    YZ = [v.target_presentpast; v.source_presentpast; C]
    Y =  [v.target_presentpast;                       C]

    p_Y  = marginal(Y, positive_measure_bins, iv)
    p_XY = marginal(XY, positive_measure_bins, iv)
    p_YZ = marginal(YZ, positive_measure_bins, iv)

    # Use base 2 for the entropy, so that we get transfer entropy in bits
    if normalise_to_tPP
        te = entropy(p_YZ, b = 2) +
            entropy(p_XY, b = 2) -
            entropy(p_Y, b = 2) -
            entropy(iv.dist[iv.nonzero_inds], b = 2)

		# Compute the transfer entropy from the target variable to itself
		# TE(Tpresent -> Tfuture | Tpast)
		#XY = [v.target_future; ]
        #te_target_on_itself =

        #te = te / (entropy(P_tF_tPP, b = 2) - entropy(P_tPP, b = 2))
    else
        te = entropy(p_YZ, b = 2) +
            entropy(p_XY, b = 2) -
            entropy(p_Y, b = 2) -
            entropy(iv.dist[iv.nonzero_inds], b = 2)
    end

    return te
end


"""
transferentropy_transferoperator_grid(
                    E::Embeddings.AbstractEmbedding,
                    ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
                    v::TransferEntropy.TEVars;
                    normalise_to_tPP = false
                    allocate_frac = 1) -> Float64

Using the transfer operator to calculate probability
distributions,  calculate transfer entropy from the embedding
`E`, given a discretization scheme controlled by `ϵ` and
information `v::TEVars` about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.

If `normalise_to_tPP = true`, then the TE estimate is normalised to
the entropy rate of the target variable `H(target_future | target_presentpast)`.
"""
function transferentropy_transferoperator_grid(
                    E::Embeddings.AbstractEmbedding,
                    ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
                    v::TransferEntropy.TEVars;
                    normalise_to_tPP = false,
                    allocate_frac = 1.0)

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
    TO = PerronFrobenius.transferoperator_binvisits(binvisits,
                        allocate_frac = allocate_frac)

    # Calculate the invariant distribution over the bins.
    invdist = left_eigenvector(TO)

    transferentropy_transferoperator_grid(
        bins_visited_by_orbit, invdist, v,
        normalise_to_tPP = normalise_to_tPP)
end


"""
    transferentropy_transferoperator_grid(E::Embeddings.AbstractEmbedding,
        ϵ::Vector{Union{Int, Float64, Vector{Float64}, Vector{Int}}},
        v::TEVars, normalise_to_tPP = false)

Compute transfer entropy over a range of bin sizes.

Using the transfer operator to calculate probability
distributions,  calculate transfer entropy from the embedding
`E`, given a discretization scheme controlled by the `ϵ`s and
information `v::TEVars` about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.

If `normalise_to_tPP = true`, then the TE estimate is normalised to
the entropy rate of the target variable, `H(target_future | target_presentpast)`.
"""
function transferentropy_transferoperator_grid(E::Embeddings.AbstractEmbedding,
        ϵ::Vector{Union{Int, Float64, Vector{Float64}, Vector{Int}}},
        v::TEVars; normalise_to_tPP = false, allocate_frac = 1.0)
    map(ϵᵢ -> transferentropy_transferoperator_grid(
        E, ϵᵢ, v;
        normalise_to_tPP = normalise_to_tPP,
        allocate_frac = allocate_frac), ϵ)
end

# Shorter alias
tetogrid = transferentropy_transferoperator_grid

#############################################################################
# One should be able to just provide some points that has been pre-embedded too
#############################################################################
"""
    transferentropy_transferoperator_grid(pts::AbstractArray{T, 2},
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
function transferentropy_transferoperator_grid(pts::AbstractArray{T, 2},
    ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
    v::TransferEntropy.TEVars;
    normalise_to_tPP = normalise_to_tPP,
    allocate_frac = allocate_frac) where T

    tetogrid(embed(pts), ϵ, v;
        normalise_to_tPP = normalise_to_tPP,
        allocate_frac = allocate_frac)
end


tetogrid = transferentropy_transferoperator_grid

export transferentropy_transferoperator_grid, tetogrid
