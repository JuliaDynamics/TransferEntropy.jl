include("helper_functions.jl")
include("get_nonempty_bins.jl")
include("joint.jl")
include("marginal.jl")

"""
    te_from_embedding(
        embedding::AbstractArray{Float64, 2},
		te_lag::Int = 1,
		binsizes::AbstractVector{Int} = vcat(1:2:20, 25:5:200, 200:10:500);
        n_reps::Int = 10,
        parallel = true,
        sparse = false
        )

Compute transfer entropy given an precomputed Markov matrix `M` for which the
relation between sourceand target time series have been embedded with the
specified `te_lag`.
"""
function te_from_mm(
        M::AbstractArray{Float64, 2},
		te_lag::Int = 1,
		binsizes::AbstractVector{Int} = vcat(1:2:20, 25:5:200, 200:10:500);
        n_reps::Int = 10,
        parallel = true,
        sparse = false)

    invmeasure, inds_nonzero_simplices = estimate_invariant_probs(M)

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
        TE_estimates = zeros(Float64, n_repetitions)

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
                point_representives(t)[inds_nonzero_simplices, :],
                invmeasure[inds_nonzero_simplices],
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

    return TEresult(embedding, lag, t, M, invmeasure, inds_nonzero_simplices,
                    binsizes, hcat(TE...).')

end
