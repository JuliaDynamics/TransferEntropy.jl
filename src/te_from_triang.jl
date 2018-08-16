include("get_nonempty_bins.jl")
include("joint.jl")
include("marginal.jl")
include("point_representatives.jl")
include("rowindexin.jl")

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
function transferentropy(
        t::StateSpaceReconstruction.Partitioning.Triangulation,
        invdist::PerronFrobenius.InvariantDistribution,
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

        positive_measure_inds = find(invdist.dist .> 1/10^12)

        # Find non-empty bins and compute their measure.
        nonempty_bins = get_nonempty_bins(
            #t.centroids[positive_measure_inds, :],
    		point_representatives(t)[positive_measure_inds, :],
    		invdist.dist[positive_measure_inds],
    		[n_bins, n_bins, n_bins]
        )


        # Compute the joint and marginal distributions.
        Pjoint = jointdist(nonempty_bins, invdist.dist[positive_measure_inds])
        Py, Pxy, Pyz, Jy, Jxy, Jyz = marginaldists(nonempty_bins, invdist.dist[positive_measure_inds])
        #Py, Pxy, Pyz,  Jy, Jxy, Jyz
        # Integrate
        for k = 1:size(Pjoint, 1)
            TE_estimates[i] += Pjoint[k] *
                log( (Pjoint[k] * Py[Jy[k]]) / (Pxy[Jxy[k]] * Pyz[Jyz[k]]) )
        end
    end

    return TE_estimates / log(2)
end


function transferentropy(t::T where {T <: StateSpaceReconstruction.Partitioning.Triangulation},
    invdist::PerronFrobenius.InvariantDistribution,
    binsizes::Vector{T} where {T<:Signed},
    n_reps::T where {T<:Signed})

    TE = SharedMatrix{Float64}(n_reps, length(binsizes))

    @sync @parallel for i in 1:length(binsizes)
        TE[:, i] = transferentropy(t, invdist, binsizes[i], n_reps)
    end

    return Array(TE)
end
