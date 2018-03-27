
"""
Create a point lying within a parent simplex, represented by a (dim+1)xdim
matrix.
"""
function childpoint(parentsimplex::Array{Float64, 2})
    dim = size(parentsimplex, 2)
    # Random linear combination coefficients
    R = rand(Float64, 1, dim + 1)

    # Normalise the coefficients so that they sum to one. We can then create
	# the new point as a convex linear combination of the vertices of the parent
	# simplex.
    R = (1 ./ sum(R, 2)) .* R

    R * parentsimplex
end

"""
Draw point representatives from the simplices of a triangulation.

Precedure:
1) Generate one point per simplex.
2) Points are generated from the interior or on the boundary of each simplex.
3) Points are drawn according to a uniform distribution.
"""
function generate_point_representives(t::Triangulation)
    dim = size(t.points, 2)
    n_simplices = size(t.simplex_inds, 1)

    # Pre-allocate array to hold the points
    point_representatives = zeros(Float64, n_simplices, dim)

    # Loop over the rows of the simplex_inds array to access all the simplices.
    for i = 1:n_simplices
        simplex = t.points[t.simplex_inds[i, :], :]
        point_representatives[i, :] = childpoint(simplex)
    end
    return point_representatives
end


"""
    te_from_triangulation_withuncertainty(
        centroids::Array{Int, 2},
        invmeasure::Vector{Float64},
        n::Int,
        repetitions::Int)

Compute transfer entropy from a triangulation (consisting of a set of simplices)
with an associated invariant distribution over the simplices. This distribution
gives the probability that trajectories - in the long term - will visit regions
of the state space occupied by each simplex.

A single transfer entropy estimate is obtained by sampling one point within each
simplex. We then overlay a rectangular grid, being regular along each
dimension, with `n` determining the number of equally sized chunks to divide
the grid into along each dimension.

The probabily of a grid cell as the weighted sum of the generated points falling
inside that bin, where the weights are the invariant probabilities associated
with the simplices to which the points belong.

Joint and marginal probability distributions are then obtained by keeping
relevant axes fixed, summing over the remaining axes.

Repeating this procedure `n_repetitions` times, we obtain a distribution of
TE estimates for this bin size.
"""
function te_from_triangulation_withuncertainty(
        t::Triangulation,
        invariantdistribution::Any,
        n::Int,
        n_repetitions::Int
    )

    # Initialise transfer entropy estimates to 0. Because the measure of the
    # bins are guaranteed to be nonnegative, transfer entropy is also guaranteed
    # to be nonnegative.
    TE_estimates_this_binsize = zeros(Float64, n_repetitions)

    for i = 1:n_repetitions
        # Represent each simplex as a single point. We can do this because
        # within the region of the state space occupied by each simplex, points
        # are indistinguishable from the point of view of the invariant measure.
        # However, when we superimpose the grid, the position of the points
        # we choose will influence the resulting marginal distributions.
        # Therefore, we have to repeat this procedure several times to get an
        # accurate transfer entropy estimate.
        #point_representatives = generate_point_representives(t)

        # Find non-empty bins and compute their measure.
        nonempty_bins, measure = get_nonempty_bins(
            #t.centroids[invariantdistribution[2], :], # if only centroids are used
            generate_point_representives(t)[invariantdistribution[2], :],
            invariantdistribution[1][invariantdistribution[2]],
            [n, n, n]
        )

        # Compute the joint distribution.
        Pjoint = jointdist(nonempty_bins, measure)

        # Compute marginal distributions.
        Py, Pxy, Pyz = marginaldists(unique(nonempty_bins, 1), measure)

        # Compute the transfer entropy.
        for j = 1:length(Pjoint)
            TE_estimates_this_binsize[i] += Pjoint[j] * log( (Pjoint[j] * Py[j]) / (Pyz[j] * Pxy[j]) )
        end
    end

    return TE_estimates_this_binsize
end


"""
Compute transfer entropy between a `source` and a target time series.
"""
function te_from_ts(source, target;
        binsizes = vcat(1:2:20, 25:5:200, 200:10:500),
        n_repetitions::Int = 10,
        te_lag::Int = 1,
        parallel = true,
        sparse = false)

    # Embed the data given the lag
    embedding = hcat(source[1:end-te_lag],
                    target[1:end-te_lag],
                    target[(1 + te_lag):end])
    embedding = invariantize_embedding(embedding, max_point_remove = 10)
    t = triang_from_embedding(Embedding(embedding))
    # Gaussian embedding

    println("Triangulating embedding that initally has ", size(t.simplex_inds, 1), " simplices...")
    #SimplexSplitting.refine_variable_k!(t, maximum(t.radii) - (maximum(t.radii)- mean(t.radii))/2)
    println("The final number of simplices is", size(t.simplex_inds, 1))
    println("Markov matrix computation ...")
    if parallel & !sparse
        M = mm_p(t)
    elseif parallel & sparse
        M = Array(mm_sparse_parallel(t))
    elseif !parallel & sparse
        M = mm_sparse(t)
    elseif !parallel & !sparse
        M = markovmatrix(t)
    end

    println("Computing invariant distribution ...")
    invmeasure, inds_nonzero_simplices = invariantdist(M)

    #te_from_triangulation_withuncertainty(t, invdist, 100, 10)

    function local_te_from_triang_withuncertainty(n::Int)


        # Initialise transfer entropy estimates to 0. Because the measure of the
        # bins are guaranteed to be nonnegative, transfer entropy is also guaranteed
        # to be nonnegative.
        TE_estimates_this_binsize = zeros(Float64, n_repetitions)

        for i = 1:n_repetitions
            # Represent each simplex as a single point. We can do this because
            # within the region of the state space occupied by each simplex, points
            # are indistinguishable from the point of view of the invariant measure.
            # However, when we superimpose the grid, the position of the points
            # we choose will influence the resulting marginal distributions.
            # Therefore, we have to repeat this procedure several times to get an
            # accurate transfer entropy estimate.
            #point_representatives = generate_point_representives(t)

            # Find non-empty bins and compute their measure.
            nonempty_bins, measure = get_nonempty_bins(
                #t.centroids[invariantdistribution[2], :],
                generate_point_representives(t)[inds_nonzero_simplices, :],
                invmeasure[inds_nonzero_simplices],
                [n, n, n]
            )

            # Compute the joint distribution.
            Pjoint = jointdist(nonempty_bins, measure)

            # Compute marginal distributions.
            Py, Pxy, Pyz = marginaldists(unique(nonempty_bins, 1), measure)

            # Compute the transfer entropy.
            for j = 1:length(Pjoint)
                TE_estimates_this_binsize[i] += Pjoint[j] * log( (Pjoint[j] * Py[j]) / (Pyz[j] * Pxy[j]) )
            end

        end

        return TE_estimates_this_binsize
    end
    TE = pmap(local_te_from_triang_withuncertainty, Progress(length(binsizes)), binsizes)

    return TEresult(embedding, lag, t, M, invmeasure, inds_nonzero_simplices, binsizes, hcat(TE...).')

end
