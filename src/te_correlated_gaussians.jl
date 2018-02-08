using Distributions
using SimplexSplitting
using InvariantDistribution
using TransferEntropy

function embed_correlated_gaussians(;npts::Int = 100, covariance::Float64 = 0.4)
    # Create an uncorrelated source and target
    dist = Normal()

    invariant_embedding_found = false
    while !invariant_embedding_found
        source = rand(dist, npts, 1)
        dest = covariance .* source[1:end] .+ (1.0 - covariance) .* rand(dist, npts, 1)
        # Embedding
        embedd = hcat(source[1:end-1], source[2:end], dest[2:end])

        if invariantset(embedd)
            invariant_embedding_found = true
            return Embedding(embedd)
        end
    end
end

embed_correlated_gaussians()

"""
Compute transfer entropy. The maximal bin size is set to the maximum simplex radius
in the triangulation.
"""
function te_correlated_gaussians_init(npts::Int = 10, covariance = 0.4)
    println("\nINITIALISING TE for correlated gaussians")
    dim = 3
    # Create an embedding from correlated gaussians related by `covariance`
    print("Generating embedding from correlated gaussians with covariance ... ", covariance)
    t_now = time_ns()
    embedding = embed_correlated_gaussians(;npts = npts, covariance = covariance)
    t_next = time_ns()
    println(" took ", (t_next - t_now)/10^9, " seconds")

    print("Triangulating embedding ...")
    t_now = time_ns()
    t = SimplexSplitting.triang_from_embedding(embedding)
    t_next = time_ns()
    println(" took ", (t_next - t_now)/10^9, " seconds")
    println("There are ", size(t.simplex_inds, 1), " simplices in the original triangulation.")

    print("Markov matrix computation ...")
    t_now = time_ns()
    M = mm_sparse(t)
    t_next = time_ns()
    print(" which contains ", nnz(M), " nonzero entries")

    println(" took ", (t_next - t_now)/10^9, " seconds")

    print("Computing invariant distribution ...")
    t_now = time_ns()
    invdist, inds = invariantdist(M)
    t_next = time_ns()
    println(" took ", (t_next - t_now)/10^9, " seconds")

    # What ranges along each dimension does the triangulation span (strictly speaking,
    # we're just considering the volume that the _centroids_ span here).
    min_r = minimum(t.radii)
    mean_r = mean(t.radii)
    max_r = maximum(t.radii)
    mins = [minimum(t.centroids[:, i]) for i in 1:dim]
    maxes = [maximum(t.centroids[:, i]) for i in 1:dim]
    ranges = [(maxes[i] - mins[i]) for i in 1:dim]

    # Compute TE
    binsizes = linspace(min_r, max_r, 100)
    @show binsizes

    TE_binsizes = zeros(length(binsizes))
    for i in 1:length(binsizes)
        nonempty_bins, measure, ranges = get_nonempty_bins_abs(t.centroids,
                                                    invdist,
                                                    [binsizes[i], binsizes[i], binsizes[i]])
        @show nonempty_bins
        # Compute joint distribution.
        joint = jointdist(nonempty_bins, measure)

        # Compute marginal distributions.
        Py, Pxy, Pyz = marginaldists(unique(nonempty_bins, 1), measure)
        # Initialise transfer entropy to 0. Because the measure of the bins
        # are guaranteed to be nonnegative, transfer entropy is also guaranteed
        # to be nonnegative.
        # Compute transfer entropy.
        for i in eachindex(joint)
            println(joint[i] .* log(joint[i] * Py[i] / (Pyz[i] * Pxy[i])))
            TE_binsizes[i] += joint[i] .* log(joint[i] * Py[i] / (Pyz[i] * Pxy[i]))
        end
    end

    @show TE_binsizes
    return TE_binsizes
end


"""
Compute transfer entropy. The maximal bin size is set to the maximum simplex radius
in the triangulation.
"""
function te_correlated_gaussians(npts::Int = 50, covariance = 0.4)
    dim = 3
    println("\nCOMPUTING TE FOR CORRELATED GAUSSIANS")

    # Create an embedding from correlated gaussians related by `covariance`
    print("Generating embedding from correlated gaussians with covariance ... ", covariance)
    t_now = time_ns()
    embedding = embed_correlated_gaussians(;npts = npts, covariance = covariance)
    t_next = time_ns()
    println(" took ", (t_next - t_now)/10^9, " seconds")



    print("Triangulating embedding ...")
    t_now = time_ns()
    t = SimplexSplitting.triang_from_embedding(embedding)
    t_next = time_ns()
    println(" took ", (t_next - t_now)/10^9, " seconds")
    println("There are ", size(t.simplex_inds, 1), " simplices in the original triangulation.")

    min_r = minimum(t.radii)
    mean_r = quantile(t.radii, 0.95)#mean(t.radii)
    max_r = maximum(t.radii)

    print("Triangulating being splitted ...")
    t_now = time_ns()
    #SimplexSplitting.refine_variable_k!(t, mean_r)
    t_next = time_ns()
    println(" took ", (t_next - t_now)/10^9, " seconds")
    println("There are ", size(t.simplex_inds, 1), " simplices in the slitted triangulation.")

    print("Markov matrix computation ...")
    t_now = time_ns()
    M = mm_sparse(t)
    t_next = time_ns()
    print(" which contains ", nnz(M), " nonzero entries")

    println(" took ", (t_next - t_now)/10^9, " seconds")


    print("Computing invariant distribution ...")
    t_now = time_ns()
    invdist, inds = invariantdist(M)
    t_next = time_ns()
    println(" took ", (t_next - t_now)/10^9, " seconds")


    # What ranges along each dimension does the triangulation span (strictly speaking,
    # we're just considering the volume that the _centroids_ span here).
    mins = [minimum(t.centroids[:, i]) for i in 1:dim]
    maxes = [maximum(t.centroids[:, i]) for i in 1:dim]
    ranges = [(maxes[i] - mins[i]) for i in 1:dim]

    # Compute TE
    binsizes = mean_r*10:(mean_r*10/100):min_r
    TE = zeros(length(binsizes))

    for i in 1:length(binsizes)
        bin_size = binsizes[i]

        nonempty_bins, measure, ranges = get_nonempty_bins_abs(t.centroids,
                                                    invdist,
                                                    [bin_size, bin_size, bin_size])
        # Compute joint distribution.
        joint = jointdist(nonempty_bins, measure)

        # Compute marginal distributions.
        Py, Pxy, Pyz = marginaldists(unique(nonempty_bins, 1), measure)

        # Initialise transfer entropy to 0. Because the measure of the bins
        # are guaranteed to be nonnegative, transfer entropy is also guaranteed
        # to be nonnegative.
        TE = zeros{Float64}(length(Py))

        # Compute transfer entropy.
        for i = 1:length(joint)
            TE[i] += joint[i] * log(joint[i] * Py[i] / (Pyz[i] * Pxy[i]))
        end
        TE_binsizes[i] = TE
    end
end
