export mutualinfo, Kraskov1, Kraskov2

# naive application of estimators in Entropies.jl
function mutualinfo(x::Vector_or_Dataset, y::Vector_or_Dataset, est::NearestNeighborEntropyEstimator; base = 2)
    X = genentropy(Dataset(x), est; base = base)
    Y = genentropy(Dataset(y), est; base = base)
    XY = genentropy(Dataset(x, y), est; base = base)
end

abstract type KNNMutualInformationEstimator <: MutualInformationEstimator end

# bias-corrected estimators from Kraskov et al. (2004)


"""
    Kraskov1(k::Int = 1; metric_x = Chebyshev(), metric_y = Chebyshev()) <: MutualInformationEstimator

The ``I^{(1)}`` nearest neighbor based mutual information estimator from 
Kraskov et al. (2004), using `k` nearest neighbors. The distance metric for 
the marginals ``x`` and ``y`` can be chosen separately, while the `Chebyshev` metric 
is always used for the `z = (x, y)` joint space.
"""
struct Kraskov1 <: KNNMutualInformationEstimator
    k::Int
    metric_x::Metric
    metric_y::Metric
    metric_z::Metric # always Chebyshev, otherwise estimator is not valid!
    
    function Kraskov1(k::Int = 1; metric_x = Chebyshev(), metric_y = Chebyshev())
        new(k, metric_x, metric_y, Chebyshev())
    end
end

"""
    Kraskov2(k::Int = 1; metric_x = Chebyshev(), metric_y = Chebyshev()) <: MutualInformationEstimator

The ``I^{(2)}(x, y)`` nearest neighbor based mutual information estimator from 
Kraskov et al. (2004), using `k` nearest neighbors. The distance metric for 
the marginals ``x`` and ``y`` can be chosen separately, while the `Chebyshev` metric 
is always used for the `z = (x, y)` joint space.
"""
struct Kraskov2 <: KNNMutualInformationEstimator
    k::Int
    metric_x::Metric
    metric_y::Metric
    metric_z::Metric # always Chebyshev, otherwise estimator is not valid!

    function Kraskov2(k::Int = 1; metric_x = Chebyshev(), metric_y = Chebyshev())
        new(k, metric_x, metric_y, Chebyshev())
    end
end

function eval_dists_to_knns!(ds, pts, knn_idxs, metric)
    @inbounds for i in 1:length(pts)
        ds[i] = evaluate(metric, pts[i], pts[knn_idxs[i]])
    end

    return ds
end

# In the Kraskov1 estimator, ϵs are the distances in the Z = (X, Y) joint space
# In the Kraskov2 estimator, ϵs are the distances in the X and Y marginal spaces
function count_within_radius!(p, x, metric, ϵs, N)
    @inbounds for i in 1:N
        ϵ = ϵs[i] / 2
        xᵢ = x[i]
        p[i] = count(evaluate(metric, xᵢ, x[j]) < ϵ for j in 1:N)
    end

    return p
end


function mutualinfo(x::Vector_or_Dataset{D1, T}, y::Vector_or_Dataset{D2, T}, est::Kraskov1; 
        base = MathConstants.e) where {D1, D2, T}
    @assert length(x) == length(y)
    z = Dataset(x, y)
    X = Dataset(x)
    Y = Dataset(y)
    N = length(z)
    
    # Common for both kraskov estimators
    tree_z = KDTree(z, est.metric_z)
    tree_x = KDTree(X, est.metric_x)
    tree_y = KDTree(Y, est.metric_y)
    
    k = est.k
    idxs_z, dists_z = knn(tree_z, z.data, k + 1, true)
    idxs_x, dists_x = knn(tree_x, X.data, k + 1, true)
    idxs_y, dists_y = knn(tree_y, Y.data, k + 1, true)

    kth_nns_z = [idx_z[k + 1] for idx_z in idxs_z]
    ϵs_z = [dz[k + 1] for dz in dists_z]
    ϵs_x = zeros(Float64, N)
    ϵs_y = zeros(Float64, N)
    eval_dists_to_knns!(ϵs_x, x, kth_nns_z, est.metric_x)
    eval_dists_to_knns!(ϵs_y, y, kth_nns_z, est.metric_y)
    
    # if the following equality holds for all points, then things are correct until this point
    #ϵ_maxes = [max(a, b) for (a, b) in zip(ϵs_x, ϵs_y)]
    #@assert all(ϵ_maxes .== ϵs_z)
    
    # Kraskov1 estimator
    nx = zeros(Int, N)
    ny = zeros(Int, N)
    count_within_radius!(nx, x, est.metric_x, ϵs_z, N)
    count_within_radius!(ny, y, est.metric_y, ϵs_z, N)

    MI = digamma(est.k) - sum(digamma.(nx .+ 1) + digamma.(ny .+ 1))/N + digamma(N) 
    
    # Kraskov uses the natural logarithm in their derivations, so need to convert in the last step
    if base != MathConstants.e
        return MI / log(base)
    else 
        return MI
    end
end


function mutualinfo(x::Vector_or_Dataset{D1, T}, y::Vector_or_Dataset{D2, T}, est::Kraskov2; 
        base = MathConstants.e) where {D1, D2, T}
    @assert length(x) == length(y)
    z = Dataset(x, y)
    X = Dataset(x)
    Y = Dataset(y)
    N = length(z)
    
    # Common for both kraskov estimators
    tree_z = KDTree(z, est.metric_z)
    tree_x = KDTree(X, est.metric_x)
    tree_y = KDTree(Y, est.metric_y)
    
    k = est.k
    idxs_z, dists_z = knn(tree_z, z.data, k + 1, true)
    idxs_x, dists_x = knn(tree_x, X.data, k + 1, true)
    idxs_y, dists_y = knn(tree_y, Y.data, k + 1, true)

    kth_nns_z = [idx_z[k + 1] for idx_z in idxs_z]
    ϵs_z = [dz[k + 1] for dz in dists_z]
    ϵs_x = zeros(Float64, N)
    ϵs_y = zeros(Float64, N)
    eval_dists_to_knns!(ϵs_x, x, kth_nns_z, est.metric_x)
    eval_dists_to_knns!(ϵs_y, y, kth_nns_z, est.metric_y)
    
    # if the following equality holds for all points, then things are correct until this point
    #ϵ_maxes = [max(a, b) for (a, b) in zip(ϵs_x, ϵs_y)]
    #@assert all(ϵ_maxes .== ϵs_z)
    
    # Kraskov2 estimator
    nx = zeros(Int, N)
    ny = zeros(Int, N)
    count_within_radius!(nx, x, est.metric_x, ϵs_x, N)
    count_within_radius!(ny, y, est.metric_y, ϵs_y, N)
    
    MI = digamma(est.k) - 1/k - sum(digamma.(nx) .+ digamma.(ny))/N + digamma(N) 
    
    # Kraskov uses the natural logarithm in their derivations, so need to convert in the last step
    if base != MathConstants.e
        return MI / log(base)
    else 
        return MI
    end
end