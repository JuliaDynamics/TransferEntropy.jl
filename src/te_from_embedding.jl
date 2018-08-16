using ProgressMeter, PmapProgressMeter, InvariantDistribution

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
function transferentropy(
        embedding::AbstractArray{Float64, 2},
		te_lag::Int = 1;
		binsizes::AbstractVector{Int} = vcat(1:2:20),
        n_reps::Int = 10,
        parallel = true,
        sparse = false,
		discrete = true, sample_uniformly = true, n_randpts = 100)

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
    t = SimplexSplitting.triang_from_embedding(SimplexSplitting.Embedding(embedding))

    #=
    # Refine triangulation until all simplices have radii less than
    `target_radius`
    =#
    max_r, mean_r = maximum(t.radii), maximum(t.radii)
    target_radius = max_r - (max_r - mean_r)/2
    SimplexSplitting.refine_variable_k!(t, target_radius)

    if discrete
		println("Transfer operator (discrete)")
        M = mm_dd2(t, n_randpts =  n_randpts, sample_randomly = !sample_uniformly)
	else
		println("Transfer operator (exact)")
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

	println("Computing left eigenvector")
    invdist = InvariantDistribution.left_eigenvector(M)

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
        TE_estimates = zeros{Float64}(n_reps)

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
	println("Transfer entropy")
    # Parallelise transfer entropy estimates over bin sizes. Add progress meter.
    TE = pmap(local_te_from_triang, Progress(length(binsizes)), binsizes)

    return embedding, t, M, invdist, TEresult(embedding, te_lag, t, M, invdist,
                    binsizes, hcat(TE...).')

end
