include("TEresult.jl")

"""
    te_correlated_gaussians(;
        binsizes = vcat(1:1:10, 12:2:200, 205:5:750, 750:10:1000, 1025:25:1500,
                        1550:50:2000, 2100:100:3000, 3200:200:5000),
        lag::Int = 1,
        n_pts::Int = 100,
        covariance = 0.4,
        parallel = true,
        sparse = true)

Compute transfer entropy for a set of correlated Gaussian source and target time series.

The transfer entropy is computed several times over a grid that varies in size according
to `binsizes` (the default values are fine to use). The lag in the transfer entropy formula
is set by `lag`, which defaults to `lag = 1`.

To run the routine in parallel (recommended, and default), start julia in the terminal as
`julia -p n_workers`. For example, if you have 4 cpu cores, start Julia with `julia -p 4`.
Setting `parallel = true` will then compute transfer entropy in parallel.

The `sparse` argument indicates whether the Markov matrix computation, which is used to
obtain the invariant probability distribution on the simplices in the  triangulated state
space, is performed on dense or sparse arrays. The sparse approach is the default.

"""
function te_correlated_gaussians(;
    binsizes = vcat(1:1:10, 12:2:200, 205:5:750, 750:10:1000, 1025:25:1500, 1550:50:2000,
                    2100:100:3000, 3200:200:5000),
    lag::Int = 1,
    n_pts::Int = 100,
    covariance = 0.4,
    parallel = true,
    sparse = false)

    # Gaussian embedding
    embedding = InvariantDistribution.invariant_gaussian_embedding(
        npts = n_pts,
        covariance = covariance,
        tau = 1
    )

    t = SimplexSplitting.triang_from_embedding(SimplexSplitting.Embedding(embedding))

    print("Triangulating embedding that initally has ",
            size(t.simplex_inds, 1), " simplices...")
    #SimplexSplitting.refine_variable_k!(t, maximum(t.radii) -
    #                                   (maximum(t.radii)- mean(t.radii))/2)
    println(" and the final number of simplices after splitting is ", size(t.simplex_inds, 1))
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


    """ Function to compute TE """
    function te_local(n::Int)

        # Find non empty bins and their measure
        nonempty_bins, measure = get_nonempty_bins(t.centroids[inds_nonzero_simplices, :],
                                                    invmeasure[inds_nonzero_simplices],
                                                    [n, n, n])

        # Compute joint distribution.
        joint = jointdist(nonempty_bins, measure)

        # Compute marginal distributions.
        Py, Pxy, Pyz = marginaldists(unique(nonempty_bins, 1), measure)

        # Initialise transfer entropy to 0. Because the measure of the bins
        # are guaranteed to be nonnegative, transfer entropy is also guaranteed
        # to be nonnegative.
        te = 0.0

        # Compute transfer entropy.
        for i = 1:length(joint)
            te += joint[i] * log(joint[i] * Py[i] / (Pyz[i] * Pxy[i]))
        end

        return te
    end

    println("Transfer entropy ...")
    TE = pmap(te_local, Progress(length(binsizes)), binsizes)

    print("Finished transfer entropy computation.")
    return embedding, lag, t, M, invmeasure, inds_nonzero_simplices, binsizes, TE
end

function te_correlated_gaussians_with_uncertainty(;
    binsizes = vcat(1:1:10, 12:2:200, 205:5:750, 750:10:1000, 1025:25:1500, 1550:50:2000,
                    2100:100:3000, 3200:200:5000),
    lag::Int = 1,
    n_pts::Int = 100,
    n_repetitions::Int = 3,
    covariance = 0.4,
    parallel = true,
    sparse = false)

    # Gaussian embedding
    embedding = InvariantDistribution.invariant_gaussian_embedding(
        npts = n_pts,
        covariance = covariance,
        tau = 1
    )

    t = SimplexSplitting.triang_from_embedding(SimplexSplitting.Embedding(embedding))

    print("Triangulating embedding that initally has ",
            size(t.simplex_inds, 1), " simplices...")
    #SimplexSplitting.refine_variable_k!(t, maximum(t.radii) -
    #                                   (maximum(t.radii)- mean(t.radii))/2)
    println(" and the final number of simplices after splitting is ", size(t.simplex_inds, 1))
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


    """ Function to compute TE """
    function te_local(n::Int)

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

    println("Transfer entropy ...")
    TE = pmap(te_local, Progress(length(binsizes)), binsizes)

    print("Finished transfer entropy computation.")
    return embedding, lag, t, M, invmeasure, inds_nonzero_simplices, binsizes, TE
end


#
# """
# Compute transfer entropy. The maximal bin size is set to the maximum simplex radius
# in the triangulation.
# """
# function te_correlated_gaussians(npts::Int = 50, covariance = 0.4)
#     dim = 3
#     println("\nCOMPUTING TE FOR CORRELATED GAUSSIANS")
#
#     # Create an embedding from correlated gaussians related by `covariance`
#     print("Generating embedding from correlated gaussians with covariance ... ", covariance)
#     t_now = time_ns()
#     embedding = embed_correlated_gaussians(;npts = npts, covariance = covariance)
#     t_next = time_ns()
#     println(" took ", (t_next - t_now)/10^9, " seconds")
#
#
#
#     print("Triangulating embedding ...")
#     t_now = time_ns()
#     t = SimplexSplitting.triang_from_embedding(embedding)
#     t_next = time_ns()
#     println(" took ", (t_next - t_now)/10^9, " seconds")
#     println("There are ", size(t.simplex_inds, 1), " simplices in the original triangulation.")
#
#     min_r = minimum(t.radii)
#     mean_r = quantile(t.radii, 0.95)#mean(t.radii)
#     max_r = maximum(t.radii)
#
#     print("Triangulating being splitted ...")
#     t_now = time_ns()
#     #SimplexSplitting.refine_variable_k!(t, mean_r)
#     t_next = time_ns()
#     println(" took ", (t_next - t_now)/10^9, " seconds")
#     println("There are ", size(t.simplex_inds, 1), " simplices in the slitted triangulation.")
#
#     print("Markov matrix computation ...")
#     t_now = time_ns()
#     M = mm_sparse(t)
#     t_next = time_ns()
#     print(" which contains ", nnz(M), " nonzero entries")
#
#     println(" took ", (t_next - t_now)/10^9, " seconds")
#
#
#     print("Computing invariant distribution ...")
#     t_now = time_ns()
#     invdist, inds = invariantdist(M)
#     t_next = time_ns()
#     println(" took ", (t_next - t_now)/10^9, " seconds")
#
#
#     # What ranges along each dimension does the triangulation span (strictly speaking,
#     # we're just considering the volume that the _centroids_ span here).
#     mins = [minimum(t.centroids[:, i]) for i in 1:dim]
#     maxes = [maximum(t.centroids[:, i]) for i in 1:dim]
#     ranges = [(maxes[i] - mins[i]) for i in 1:dim]
#
#     # Compute TE
#     binsizes = mean_r*10:(mean_r*10/100):min_r
#     TE = zeros(length(binsizes))
#
#     for i in 1:length(binsizes)
#         bin_size = binsizes[i]
#
#         nonempty_bins, measure, ranges = get_nonempty_bins_abs(t.centroids,
#                                                     invdist,
#                                                     [bin_size, bin_size, bin_size])
#         # Compute joint distribution.
#         joint = jointdist(nonempty_bins, measure)
#
#         # Compute marginal distributions.
#         Py, Pxy, Pyz = marginaldists(unique(nonempty_bins, 1), measure)
#
#         # Initialise transfer entropy to 0. Because the measure of the bins
#         # are guaranteed to be nonnegative, transfer entropy is also guaranteed
#         # to be nonnegative.
#         TE = zeros{Float64}(length(Py))
#
#         # Compute transfer entropy.
#         for i = 1:length(joint)
#             TE[i] += joint[i] * log(joint[i] * Py[i] / (Pyz[i] * Pxy[i]))
#         end
#         TE_binsizes[i] = TE
#     end
# end
