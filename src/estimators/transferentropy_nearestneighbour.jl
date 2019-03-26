import Distances: Metric, Chebyshev, pairwise, colwise
import NearestNeighbors: KDTree, knn
import SpecialFunctions: digamma
import StaticArrays: SVector, MVector 

"""
    transferentropy(points, v::TEVars, estimator::NNEstimator(); 
        metric = Chebyshev(), k1::Int, k2::Int)

Compute transfer entropy decomposed as the sum of mutual informations,
using an adapted version of the Kraskov estimator for mutual information [1].

## Arguments
- `points`: The set of points representing the embedding for which to compute
    transfer entropy. Must be provided as an vector of state vectors, or an 
    array of size `dim`-by-`n` points.
- `k1`: The number of nearest neighbours for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `k2`: The number of nearest neighbours for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `v`: A `TEVars` instance, indicating which variables of the embedding should
    be grouped as what when computing the marginal entropies that go into the
    transfer entropy expression.

## Keyword arguments
- `metric`: The distance metric. Must be a valid metric from `Distances.jl`.
- `b`: The transfer entropy obtained is scaled by `log(b)`. This corresponds 
        to taking the logarithm to the base `b` if computing transfer entropy 
        over a partition using, for example, a visitation frequency approach.

# References
1. Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
    mutual information." Physical review E 69.6 (2004): 066138.
"""
function transferentropy end

function transferentropy(points::Vector{T}, vars::TEVars, estimator::NNEstimator; 
    metric::Metric = Chebyshev(), b = 2, k1::Int = 2, k2::Int = 3) where {T <: Union{Vector, SVector, MVector}}

    transferentropy_kraskov(hcat(points...,), vars, estimator, 
        metric = metric, b = b, k1 = k2, k2 = k2)
end


function marginal_nearestneighbours(points, dists_to_kth)
    D = pairwise(Chebyshev(), points, dims = 2)

    npts = size(points, 2)
    N = zeros(Int, npts)

    for i = 1:npts
        N[i] = sum(D[:, i] .<= dists_to_kth[i])
    end

    return N
end


function transferentropy_kraskov(points::AbstractArray{T, 2}, vars::TEVars, estimator::NNEstimator; 
        metric::Metric = Chebyshev(), b = 2, k1::Int = 2, k2::Int = 3) where T

    # Make sure that the array contains points as columns.
    if size(points, 1) > size(points, 2)
        error("The dimension of the dataset exceeds the number of points.")
    end
    # The total number of points
    N = size(points, 2)

    # Create some dummy variable names to avoid cluttering the code too much
    X = vars.target_future
    Y = vars.target_presentpast
    Z = vcat(vars.source_presentpast, vars.conditioned_presentpast)
    XY = vcat(X, Y)
    YZ = vcat(Y, Z)

    pts_X = points[X, :]
    pts_Y = points[Y, :]
    pts_XY = points[XY, :]
    pts_YZ = points[YZ, :]
    pts_XYZ = points

    # Create trees to search for nearest neighbors
    tree_XYZ = KDTree(pts_XYZ, metric)
    tree_XY = KDTree(pts_XY, metric)

    # Find the k nearest neighbors to all of the points in each of the trees
    idxs_XYZ, dists_XYZ = knn(tree_XYZ, pts_XYZ, k1, true)
    idxs_XY, dists_XY   = knn(tree_XY,  pts_XY, k2, true)

    # In each of the trees, find the index of the k-th nearest neighbor to all
    # of the points.
    kth_NN_idx_XYZ = [idx[k1] for idx in idxs_XYZ]
    kth_NN_idx_XY  = [idx[k2] for idx in idxs_XY]

    # Distances between points in the XYZ and XY spaces and their
    # kth nearest neighbour, along marginals X, YZ (for the XYZ space)
    # and X, Y (for the XY space).
    ϵ_XYZ_X = colwise(metric, pts_X, pts_X[:, kth_NN_idx_XYZ])
    ϵ_XYZ_YZ = colwise(metric, pts_YZ, pts_YZ[:, kth_NN_idx_XYZ])
    ϵ_XY_X = colwise(metric, pts_X, pts_X[:, kth_NN_idx_XY])
    ϵ_XY_Y = colwise(metric, pts_Y, pts_Y[:, kth_NN_idx_XY])

    NXYZ_X  = marginal_nearestneighbours(pts_X,  ϵ_XYZ_X)
    NXYZ_YZ = marginal_nearestneighbours(pts_YZ, ϵ_XYZ_YZ)
    NXY_X   = marginal_nearestneighbours(pts_X,  ϵ_XY_X)
    NXY_Y   = marginal_nearestneighbours(pts_Y,  ϵ_XY_Y)

    te = sum(digamma.(NXY_X) + digamma.(NXY_Y) -
        digamma.(NXYZ_X) - digamma.(NXYZ_YZ)) / N
    
    te / log(b)
end

