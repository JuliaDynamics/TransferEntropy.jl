export transferentropy, NearestNeighborMI

import Distances: Metric, Chebyshev
import ..TEVars
import .._transferentropy 
import DelayEmbeddings: Dataset 
import StaticArrays: SVector, MVector
import SpecialFunctions: digamma 
import NearestNeighbors: knn

"""
    NearestNeighborMI(k1::Int = 2, k2::Int = 3, metric::Metric = Chebyshev, b::Number)

A transfer entropy estimator using counting of nearest neighbors nearest neighbors 
to estimate mutual information over an appropriate generalized delay reconstruction of 
the input data (MI estimation method from Kraskov et al. (2004)[^Kraskov2004], as implemented 
for transfer entropy in Diego et al. (2019)[^Diego2019]).

## Fields 

- **`k1::Int = 2`**: The number of nearest neighbors for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbors, say `k1 = k2 = 4`, will suffice.
- **`k2::Int = 3`**: The number of nearest neighbors for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbors, say `k1 = k2 = 4`, will suffice.
- **`metric::Metric = Chebyshev()`**: The metric used for distance computations.
- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).

## References

[^Kraskov2004]: Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating mutual information." Physical review E 69.6 (2004): 066138.
[^Diego2019]: Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
@Base.kwdef struct NearestNeighborMI{M <: Metric} <: NearestNeighborTransferEntropyEstimator{M}
    k1::Int = 2
    k2::Int = 3
    metric::M = Chebyshev()
    b::Number = 2
end

function Base.show(io::IO, estimator::NearestNeighborMI)
    k1 = estimator.k1
    k2 = estimator.k2
    metric = estimator.metric
    b = estimator.b
    s = "$(typeof(estimator))(b=$(b), k1=$(k1), k2=$(k2), metric=$(typeof(metric)))"
    print(io, s)
end

function marginal_NN(points::VSV, dists_to_kth)
    D = pairwise(points, Chebyshev())
    
    N = zeros(Int, length(points))

    @inbounds for i = 1:length(points)
        N[i] = sum(D[:, i] .<= dists_to_kth[i])
    end

    return N
end

const SVV{N, T} = Union{Vector{<:SVector{N, T}}, Vector{<:MVector{N, T}}} where {N, T}

"""
    get_marginal_pts(t::Type{NearestNeighborMI}, pts::SVV{N, T}, v::TEVars) → (pts_X, pts_Y, pts_XY, pts_YZ, pts_XYZ)

Subset relevant marginals of `pts` for transfer entropy computation, as given by `vars`. 
These marginals are used for the `NearestNeighborMI` estimator.
"""
function get_marginal_pts(t::NearestNeighborMI, pts::SVV{N, T}, v::TEVars) where {N, T}

     # Create some dummy variable names to avoid cluttering the code too much
    X = v.𝒯
    Y = v.T
    Z = vcat(v.S, v.C)
    XY = vcat(X, Y)
    YZ = vcat(Y, Z)
    
    nX = length(X)
    nY = length(Y)
    nZ = length(Z)
    nXY = nX + nY
    nYZ = nY + nZ
    
    pts_X = [SVector{nX, T}(pt[X]) for pt in pts]
    pts_Y = [SVector{nY, T}(pt[Y]) for pt in pts]
    pts_XY = [SVector{nXY, T}(pt[XY]) for pt in pts]
    pts_YZ = [SVector{nYZ, T}(pt[YZ]) for pt in pts]
    pts_XYZ = pts

    return pts_X, pts_Y, pts_XY, pts_YZ, pts_XYZ
end

function get_marginal_pts(t::NearestNeighborMI, pts::Vector{Vector{T}}, vars::TEVars) where T
    get_marginal_pts(t, Dataset(pts), vars)
end

function get_marginal_pts(t::NearestNeighborMI, pts::Dataset, v::TEVars)
    
     # Create some dummy variable names to avoid cluttering the code too much
    X = v.𝒯
    Y = v.T
    Z = vcat(v.S, v.C)
    XY = vcat(X, Y)
    YZ = vcat(Y, Z)
    
    nX = length(X)
    nY = length(Y)
    nZ = length(Z)
    nXY = nX + nY
    nYZ = nY + nZ
    
    pts_X = pts[:, X]
    pts_Y = pts[:, Y]
    pts_XY = pts[:, XY]
    pts_YZ = pts[:, YZ]
    pts_XYZ = pts

    return pts_X, pts_Y, pts_XY, pts_YZ, pts_XYZ
    
end

function _transferentropy(pts, v::TEVars, estimator::NearestNeighborMI{M}) where M
    
    metric = estimator.metric
    b = estimator.b
    k1 = estimator.k1
    k2 = estimator.k2
    
    npts = length(pts)

    pts_X, pts_Y, pts_XY, pts_YZ, pts_XYZ = get_marginal_pts(estimator, pts, v)

    # Create trees to search for nearest neighbors
    tree_XYZ = KDTree(pts_XYZ, metric)
    tree_XY = KDTree(pts_XY, metric)
    # Find the k nearest neighbors to all of the points in each of the trees.
    # We sort the points according to their distances, because we want precisely
    # the k-th nearest neighbor.
    knearest_XYZ = [knn(tree_XYZ, pt, k1, true) for pt in pts_XYZ]
    knearest_XY   = [knn(tree_XY, pt, k2, true) for pt in pts_XY]
    
    # In each of the trees, find the index of the k-th nearest neighbor to all
    # of the points.
    kth_NN_idx_XYZ = [ptnbrs[1][k1] for ptnbrs in knearest_XYZ]
    kth_NN_idx_XY  = [ptnbrs[1][k2] for ptnbrs in knearest_XY]
    
    # Distances between points in the XYZ and XY spaces and their
    # kth nearest neighbor, along marginals X, YZ (for the XYZ space)
    ϵ_XYZ_X = find_dists(pts_X, pts_X[kth_NN_idx_XYZ], metric)    
    ϵ_XYZ_YZ = find_dists(pts_YZ, pts_YZ[kth_NN_idx_XYZ], metric)
    ϵ_XY_X = find_dists(pts_X, pts_X[kth_NN_idx_XY], metric)
    ϵ_XY_Y = find_dists(pts_Y, pts_Y[kth_NN_idx_XY], metric)
    
    NXYZ_X  = marginal_NN(pts_X,  ϵ_XYZ_X)
    NXYZ_YZ = marginal_NN(pts_YZ, ϵ_XYZ_YZ)
    NXY_X   = marginal_NN(pts_X,  ϵ_XY_X)
    NXY_Y   = marginal_NN(pts_Y,  ϵ_XY_Y)
    
    # Compute transfer entropy
    te = sum(digamma.(NXY_X) + digamma.(NXY_Y) -
            digamma.(NXYZ_X) - digamma.(NXYZ_YZ)) / npts
    
    # Convert from nats to to the desired unit (b^x = e^1 => 1/ln(b))
    return te / log(b)
end