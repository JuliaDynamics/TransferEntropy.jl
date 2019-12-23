import Distances: pairwise
import DynamicalSystemsBase: AbstractDataset, Dataset
import StaticArrays: SVector

const VSV = Union{AbstractDataset, Vector{<:SVector}}


function pairwise(x::VSV, y::VSV, metric::Metric)
    length(x) == length(y) || error("Lengths of input data does not match")

    dists = zeros(eltype(eltype(x)), length(x), length(y))
    @inbounds for j in 1:length(y)
        for i in 1:length(x)
            dists[i, j] = evaluate(metric, x[i], y[j])
        end
    end
    return dists
end

function pairwise(x::VSV, metric::Metric)
    d = zeros(eltype(eltype(x)), length(x), length(x))
    @inbounds for j in 1:length(x)
        for i in 1:length(x)
            dists[i, j] = evaluate(metric, x[i], x[j])
        end
    end
    return dists
end

function find_dists(x::VSV, y::VSV, metric::Metric)
    length(x) == length(y) || error("Lengths of input data does not match")
    dists = zeros(eltype(eltype(x)), length(x))
    
    @inbounds for i = 1:length(x)
        dists[i] = evaluate(metric, x[i], y[i])
    end
    
    return dists
end

function marginal_NN(points::VSV, dists_to_kth)
    D = pairwise(points, Chebyshev())
    
    N = zeros(Int, length(points))

    @inbounds for i = 1:length(points)
        N[i] = sum(D[:, i] .<= dists_to_kth[i])
    end

    return N
end

import TransferEntropy.transferentropy 
import DelayEmbeddings: KDTree
KDTree(D::CustomReconstruction, metric::Metric = Euclidean()) = KDTree(D.reconstructed_pts, metric)

function transferentropy(points::AbstractDataset, vars::TEVars, estimator::NearestNeighbourMI)
    
    metric = estimator.metric
    b = estimator.b
    k1 = estimator.k1
    k2 = estimator.k2
    
    npts = length(points)

    # Create some dummy variable names to avoid cluttering the code too much
    X = vars.target_future
    Y = vars.target_presentpast
    Z = vcat(vars.source_presentpast, vars.conditioned_presentpast)
    XY = vcat(X, Y)
    YZ = vcat(Y, Z)

    pts_X = points[:, X]
    pts_Y = points[:, Y]
    pts_XY = points[:, XY]
    pts_YZ = points[:, YZ]
    pts_XYZ = points
    
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
    # kth nearest neighbour, along marginals X, YZ (for the XYZ space)
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

function transferentropy(points::CustomReconstruction, vars::TEVars, estimator::NearestNeighbourMI)
    transferentropy(points.reconstructed_pts, vars, estimator)
end

