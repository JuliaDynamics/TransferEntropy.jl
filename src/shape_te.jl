
"""
Compute the transfer entropy resulting only from the geometry of the reconstructed
attractor. How? Assign uniformly distributed states on the volumes of the
reconstructed state space with nonzero measure.
"""
function shape_te(inds_unique_bins::Array{Int, 2})
   dim = size(inds_unique_bins, 2)
   n_nonempty_bins = size(inds_unique_bins, 1)

   n_Y = marginal_multiplicity(inds_unique_bins[:, 2])
   n_XY = marginal_multiplicity(inds_unique_bins[:, 1:2])
   n_YZ = marginal_multiplicity(inds_unique_bins[:, 2:3])

   p_Y = n_Y / n_nonempty_bins
   p_XY = n_XY / n_nonempty_bins
   p_YZ = n_YZ / n_nonempty_bins

   # Transfer entropy as the sum of the marginal entropies
   ((nat_entropy(p_YZ) + nat_entropy(p_XY) - nat_entropy(p_Y)) / n_nonempty_bins) / log(2)
end

"""
Compute the shape transfer entropy from a triangulation with an associated invariant probability distribution.
"""
function shape_te(
        t::Triangulation,
        invdist::InvariantDistribution.InvDist,
        n_bins::Int
    )

    positive_measure_inds = find(invdist.dist .> 1/10^12)
    nonempty_bins = get_nonempty_bins(
        point_representatives(t)[positive_measure_inds, :],
    	invdist.dist[positive_measure_inds],
    	[n_bins, n_bins, n_bins]
    )

    shape_te(nonempty_bins)
end

"""
Compute the shape transfer entropy from a triangulation with an associated invariant probability distribution.
"""
function shape_te(t::Triangulation,
    invdist::InvariantDistribution.InvDist,
    binsizes::Vector{Int})

    TE = SharedVector{Float64}(length(binsizes))

    @sync @parallel for i in 1:length(binsizes)
        TE[i] = shape_te_from_triang(t, invdist, binsizes[i])
    end

    return Array(TE)
end

"""
Compute the shape transfer entropy from a triangulation with an associated invariant probability distribution.
"""
function shape_te(t::Triangulation,
      invdist::InvariantDistribution.InvDist,
      binsizes::Range{Int})

    TE = SharedVector{Float64}(length(binsizes))

    @sync @parallel for i in 1:length(binsizes)
        TE[i] = shape_te_from_triang(t, invdist, binsizes[i])
    end

    return Array(TE)
end
