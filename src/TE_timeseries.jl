include("te_from_triangulation_with_uncertainty.jl")

"""
Compute transfer entropy between `source_timeseries` and `target_timeseries`.

"""
function TE_timeseries(source::Vector{Float64}, target::Vector{Float64}; lag::Int = 1, binsizes = vcat(1:1:10, 12:2:200, 205:5:750, 750:10:1000, 1025:25:1500, 1550:50:2000, 2100:100:3000, 3200:200:5000), npts::Int = 100, parallel = true, sparse = true)

    assert(lag != 0)

    # Embed time series
    emb = hcat(source[1:end-lag], target[1:end-lag], target[1+lag:end])


    # Make sure embedding is invariant
    emb = invariantize_embedding(emb)
    if isempty(emb)
        error("Embedding could not be invariantized.")
        return
    end
    t = SimplexSplitting.triang_from_embedding(SimplexSplitting.Embedding(emb))

    println("Triangulating embedding that initally has ", size(t.simplex_inds, 1), " simplices...")
    #SimplexSplitting.refine_variable_k!(t, maximum(t.radii) - (maximum(t.radii)- mean(t.radii))/2)
    println("The final number of simplices is", size(t.simplex_inds, 1))
    println("Markov matrix computation ...")
    if parallel & sparse
        M = Array(mm_sparse_parallel(t))
    elseif parallel & !sparse
        M = mm_p(t)
    else
        M = mm_sparse(t)
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
    return TEresult(emb, lag, t, M, invmeasure, inds_nonzero_simplices, binsizes, TE)
end



"""
Compute transfer entropy between `source_timeseries` and `target_timeseries`.

"""
function TE_timeseries_withuncertainty(source::Vector{Float64}, target::Vector{Float64}; lag::Int = 1, n_repetitions = 10, binsizes = vcat(1:1:10, 12:2:200, 205:5:750, 750:10:1000, 1025:25:1500, 1550:50:2000, 2100:100:3000, 3200:200:5000), parallel = true, sparse = false)

    assert(lag != 0)

    # Embed time series
    emb = hcat(source[1:end-lag], target[1:end-lag], target[1+lag:end])


    # Make sure embedding is invariant
    emb = invariantize_embedding(emb)
    if isempty(emb)
        error("Embedding could not be invariantized.")
        return
    end
    t = SimplexSplitting.triang_from_embedding(SimplexSplitting.Embedding(emb))

    println("Triangulating embedding that initally has ", size(t.simplex_inds, 1), " simplices...")
    #SimplexSplitting.refine_variable_k!(t, maximum(t.radii) - (maximum(t.radii)- mean(t.radii))/2)
    println("The final number of simplices is ", size(t.simplex_inds, 1))
    println("Markov matrix computation ...")
    if parallel & sparse
        M = Array(mm_sparse_parallel(t))
    elseif parallel & !sparse
        M = mm_p(t)
    else
        M = mm_sparse(t)
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


    println("Transfer entropy ...")
    TE = hcat(pmap(te_local, Progress(length(binsizes)), binsizes)...).'
    print("Finished transfer entropy computation.")
    return TEresult(emb, lag, t, M, invmeasure, inds_nonzero_simplices, binsizes, TE)
end
