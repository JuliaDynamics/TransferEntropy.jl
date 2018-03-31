include("get_nonempty_bins.jl")
include("joint.jl")
include("marginal.jl")
include("point_representatives.jl")

"""
    te_from_triang(
        t::Triangulation,
        invariantdistribution::Any,
        n_bins::Int,
        n_reps::Int
        )

Compute transfer entropy from a triangulation `t` with an associated invariant
distribution over the simplices. This distribution
gives the probability that trajectories - in the long term - will visit regions
of the state space occupied by each simplex.

A single transfer entropy estimate is obtained by sampling one point within each
simplex. We then overlay a rectangular grid, being regular along each
dimension, with `n_bins` determining the number of equally sized chunks to divide
the grid into along each dimension.

The probabily of a grid cell as the weighted sum of the generated points falling
inside that bin, where the weights are the invariant probabilities associated
with the simplices to which the points belong.

Joint and marginal probability distributions are then obtained by keeping
relevant axes fixed, summing over the remaining axes.

Repeating this procedure `n_reps` times, we obtain a distribution of
TE estimates for this bin size.
"""
function te_from_triang(
        t::Triangulation,
        invdist::InvariantDistribution.InvDist,
        n_bins::Int,
        n_reps::Int
    )

    # Initialise transfer entropy estimates to 0. Because the measure of the
    # bins are guaranteed to be nonnegative, transfer entropy is also guaranteed
    # to be nonnegative.
    TE_estimates = zeros(Float64, n_reps)

    for i = 1:n_reps
        # Represent each simplex as a single point. We can do this because
        # within the region of the state space occupied by each simplex, points
        # are indistinguishable from the point of view of the invariant measure.
        # However, when we superimpose the grid, the position of the points
        # we choose will influence the resulting marginal distributions.
        # Therefore, we have to repeat this procedure several times to get an
        # accurate transfer entropy estimate.

        # Find non-empty bins and compute their measure.
        nonempty_bins, measure = get_nonempty_bins(
    		point_representatives(t)[invdist.nonzero_inds, :],
    		invdist.dist[invdist.nonzero_inds],
    		[n_bins, n_bins, n_bins]
        )

        # Compute the joint and marginal distributions.
        Pjoint = jointdist(nonempty_bins, measure)
        Py, Pxy, Pyz = marginaldists(unique(nonempty_bins, 1), measure)

        # Integrate
        for j = 1:length(Pjoint)
            TE_estimates[i] += Pjoint[j] * log( (Pjoint[j] * Py[j]) /
                                                (Pyz[j] * Pxy[j]) )
        end
    end

    return TE_estimates
end
