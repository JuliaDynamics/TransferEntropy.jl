using Distances: Chebyshev

export Kraskov1

"""
    Kraskov1 <: MutualInformation
    Kraskov1(k::Int = 1; metric_x = Chebyshev(), metric_y = Chebyshev())

The ``I^{(1)}`` nearest neighbor based mutual information estimator from
Kraskov et al. (2004), using `k` nearest neighbors. The distance metric for
the marginals ``x`` and ``y`` can be chosen separately, while the `Chebyshev` metric
is always used for the `z = (x, y)` joint space.
"""
Base.@kwdef struct Kraskov1{MX, MY, MZ, B} <: MutualInformation
    k::Int
    metric_x::MX
    metric_y::MY
    metric_z::MZ # always Chebyshev, otherwise estimator is not valid!
    base::B

    function Kraskov1(; k::Int = 1, metric_x::MX = Chebyshev(), metric_y::MY = Chebyshev(),
            base::B = ℯ) where {MX, MY, B}
        metric_z = Chebyshev()
        new{MX, MY, typeof(metric_z), B}(k, metric_x, metric_y, Chebyshev(), base)
    end
end


function mutualinfo(est::Kraskov1,
        x::Vector_or_Dataset{D1, T}, y::Vector_or_Dataset{D2, T}) where {D1, D2, T}
    @assert length(x) == length(y)
    (; k, metric_x, metric_y, metric_z, base) = est

    z = Dataset(x, y)
    N = length(z)

    tree_z = KDTree(z, metric_z)
    idxs_z, dists_z = bulksearch(tree_z, z, NeighborNumber(k + 1))

    kth_nns_z = [idx_z[k + 1] for idx_z in idxs_z]
    ϵs_z = [dz[k + 1] for dz in dists_z]
    ϵs_x = zeros(Float64, N)
    ϵs_y = zeros(Float64, N)
    eval_dists_to_knns!(ϵs_x, x, kth_nns_z, metric_x)
    eval_dists_to_knns!(ϵs_y, y, kth_nns_z, metric_y)

    # if the following equality holds for all points, then things are correct until this point
    #ϵ_maxes = [max(a, b) for (a, b) in zip(ϵs_x, ϵs_y)]
    #@assert all(ϵ_maxes .== ϵs_z)
    nx = zeros(Int, N)
    ny = zeros(Int, N)
    count_within_radius!(nx, x, est.metric_x, ϵs_z, N)
    count_within_radius!(ny, y, est.metric_y, ϵs_z, N)

    MI = digamma(est.k) -
        sum(digamma.(nx .+ 1) +
        digamma.(ny .+ 1)) / N +
        digamma(N)

    # Kraskov uses the natural logarithm (nats) in their derivation. Convert to target unit.
    return MI / log(base, ℯ)
end
