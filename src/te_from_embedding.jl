using ProgressMeter, PmapProgressMeter

include("get_nonempty_bins.jl")
include("joint.jl")
include("marginal.jl")
include("TEresult.jl")
include("point_representatives.jl")
include("rowindexin.jl")

"""
    te_from_embedding(
        embedding::AbstractArray{Float64, 2},
		te_lag::Int = 1,
		binsizes::AbstractVector{Int} = vcat(1:2:20, 25:5:200, 200:10:500);
        n_reps::Int = 10,
        parallel = true,
        sparse = false
        )

Compute transfer entropy given an embedding in which the relation between source
and target time series have already been embedded with the specified `te_lag`.
range of `binsizes`
"""
function te_from_embedding(
        embedding::AbstractArray{Float64, 2},
		te_lag::Int = 1;
		binsizes::AbstractVector{Int} = vcat(1:2:20, 25:5:200, 200:10:500),
        n_reps::Int = 10,
        parallel = true,
        sparse = false,
		discrete = false, sample_uniformly = true,
		n_randpts = 100)

    #=
    # Embed the time series for transfer entropy estimation given the provided a
    # transfer entropy estimation lag `te_lag` and make sure the embedding is
    # invariant. This is done by moving point in the embedding corresponding
    # to the last time index towards the embeddings' center until the point
    # lies inside the convex hull of all preceding points. This ensures that
    # the embedding invariant under the action of the forward-time map we apply
    # to estimate the transfer operator.
    =#

    embedding = InvariantDistribution.invariantize_embedding(embedding, max_point_remove = 10)

    # Triangulate
    t = triang_from_embedding(Embedding(embedding))

    #=
    # Refine triangulation until all simplices have radii less than
    `target_radius`
    =#
    max_r, mean_r = maximum(t.radii), maximum(t.radii)
    target_radius = max_r - (max_r - mean_r)/2
    SimplexSplitting.refine_variable_k!(t, target_radius)

    if discrete
        M = mm_dd2(t, n_randpts =  n_randpts, sample_randomly = !sample_uniformly)
	else
        if parallel & !sparse
            M = mm_p(t)
        elseif parallel & sparse
            M = Array(mm_sparse_parallel(t))
        elseif !parallel & sparse
            M = mm_sparse(t)
        elseif !parallel & !sparse
            M = markovmatrix(t)
        end
    end

    invdist = estimate_invdist(M)

    """
        local_te_from_triang(n_bins::Int)

    Internal version of `te_from_triang` from the file `te_from_triang.jl` which
    only takes the number of bins `n_bins::Int` as an input. This way, we can
    easily apply the `pmap` function and parallelise transfer entropy estiamtion
    over bin sizes.
    """
    function local_te_from_triang(n_bins::Int)
        #=
        # Initialise transfer entropy estimates to 0. Because the measure of the
        # bins are guaranteed to be nonnegative, transfer entropy is also
        # guaranteed to be nonnegative.
        =#
        TE_estimates = zeros(Float64, n_reps)

        for i = 1:n_reps
            #=
            # Represent each simplex as a single point. We can do this because
            # within the region of the state space occupied by each simplex,
            # points are indistinguishable from the point of view of the
            # invariant measure. However, when we superimpose the grid, the
            # position of the points we choose will influence the resulting
            # marginal distributions. Therefore, we have to repeat this
            # procedure several times to get an accurate transfer entropy
            # estimate.
            =#

            # Find the indices of the non-empty bins and compute their measure.
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

    # Parallelise transfer entropy estimates over bin sizes. Add progress meter.
    TE = pmap(local_te_from_triang, Progress(length(binsizes)), binsizes)

    return embedding, t, M, invdist, TEresult(embedding, te_lag, t, M, invdist,
                    binsizes, hcat(TE...).')

end
