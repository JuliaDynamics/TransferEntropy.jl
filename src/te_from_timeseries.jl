using PmapProgressMeter
using ProgressMeter

"""
    te_from_timeseries(
        source::Vector{Float64},
        target::Vector{Float64};
        binsizes = vcat(1:1:10, 12:2:200, 205:5:750, 750:10:1000, 1025:25:1500, 1550:50:2000, 2100:100:3000, 3200:200:5000),
        lag::Int = 1,
        npts::Int = 100,
        covariance = 0.4,
        parallel = false,
        sparse = false)

Compute transfer entropy between a `source` time series and a `target` time series. The transfer entropy is computed several times over a grid that varies in size according to `binsizes` (the default values are fine to use). The lag in the transfer entropy formula is set by `lag`, which defaults to `lag = 1`.

To run the routine in parallel (recommended, and default), start julia in the terminal as `julia -p n_workers`. For example, if you have 4 cpu cores, start Julia with `julia -p 4`. Setting `parallel = true` will then compute transfer entropy in parallel.

The `sparse` argument indicates whether the Markov matrix computation, which is used to obtain the invariant probability distribution on the simplices in the  triangulated state space, is performed on dense or sparse arrays. The sparse approach is the default.

"""
function te_from_timeseries(
    source::Vector{Float64},
    target::Vector{Float64};
    binsizes = vcat(1:1:10, 12:2:200, 205:5:750, 750:10:1000, 1025:25:1500, 1550:50:2000, 2100:100:3000, 3200:200:5000),
    lag::Int = 1,
    parallel = true,
    sparse = true)

    # Embed the data given the lag
    embedding = hcat(source[1:end-lag], target[1:end-lag], target[1+lag:end])
    embedding = InvariantDistribution.invariantize_embedding(embedding, max_point_remove = 10)
    t = triang_from_embedding(Embedding(embedding))
    # Gaussian embedding

    println("Triangulating embedding that initally has ", size(t.simplex_inds, 1), " simplices...")
    #SimplexSplitting.refine_variable_k!(t, maximum(t.radii) - (maximum(t.radii)- mean(t.radii))/2)
    println("The final number of simplices is", size(t.simplex_inds))
    println("Markov matrix computation ...")
    if parallel & sparse
        M = mm_p(t)
    elseif parallel & !sparse
        M = Array(mm_sparse_parallel(t))
    else
        M = mm_sparse(t)
    end

    println("Computing invariant distribution ...")
    invmeasure, inds_nonzero_simplices = invariantdist(M)
    TE = Vector{Float64}(length(binsizes))

    count = 0
    println("Transfer entropy ...")
    for binsize in binsizes
        count +=1
        println("Bin size ", binsize)
        te = te_from_triangulation(t.centroids, invmeasure, binsize)
        TE[count] = te
    end
    print("Finished transfer entropy computation.")
    return binsizes, TE
    #return TEresult(e, lag, t, M, invmeasure, inds_nonzero_simplices, binsizes, TE)
end


"""
Compute transfer entropy between `source_timeseries` and `target_timeseries`.

"""
function te_from_timeseries_parallel(; binsizes = vcat(1:1:10, 12:2:200, 205:5:750, 750:10:1000, 1025:25:1500, 1550:50:2000, 2100:100:3000, 3200:200:5000), lag::Int = 1, npts::Int = 100, covariance = 0.4, parallel = false, sparse = false)

    # Gaussian embedding
    e = InvariantDistribution.invariant_gaussian_embedding(npts = npts, covariance = covariance, tau = 1)

    t = SimplexSplitting.triang_from_embedding(SimplexSplitting.Embedding(e))

    println("Triangulating embedding that initally has ", size(t.simplex_inds, 1), " simplices...")
    #SimplexSplitting.refine_variable_k!(t, maximum(t.radii) - (maximum(t.radii)- mean(t.radii))/2)
    println("The final number of simplices is", size(t.simplex_inds))
    println("Markov matrix computation ...")
    if parallel & sparse
        M = mm_p(t)
    elseif parallel & !sparse
        M = Array(mm_sparse_parallel(t))
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
    return TEresult(e, lag, t, M, invmeasure, inds_nonzero_simplices, binsizes, TE)
end


function draw_statespace_point(t::Triangulation, invdist::Vector{Float64})

end
