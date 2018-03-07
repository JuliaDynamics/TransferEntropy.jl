using Distributions
using InvariantDistribution
using SimplexSplitting
using TransferEntropy

function te_binsizes_ex(binsizes; npts::Int = 100, covariance = 0.4)
    # Create an uncorrelated source and target
    dist = Normal()

    source = rand(dist, npts, 1)
    dest = covariance .* source[1:end] .+ (1.0 - covariance) .* rand(dist, npts, 1)

    # Embedding
    e = hcat(source[1:end-1], source[2:end], dest[2:end])

    println("Embedding: is it invariant?")
    if invariantset(e)
      println("\tThe embedding forms an invariant set. Continuing.")
    else
      println("\tThe embedding does not form an invariant set. Returning zeros.")
      return zeros(length(binsizes))
    end
    println("Triangulating embedding ...")
    t = SimplexSplitting.triang_from_embedding(Embedding(e))
    println("Triangulating being splitted ...")
    SimplexSplitting.refine_variable_k!(t, (maximum(t.radii) - mean(t.radii))/2)
    
    println("Markov matrix computation ...")
    M = mm_sparse(t)

    println("Computing invariant distribution ...")
    invmeasure, inds_nonzero_simplices = invariantdist(M)
    TE = zeros(length(binsizes))

    count = 0
    println("Transfer entropy ...")
    for binsize in binsizes
        count +=1
        println("Bin size ", binsize)
        te = te_from_triangulation(t.centroids, invmeasure, binsize)
        TE[count] = te
    end
    println()
    return TE
end
