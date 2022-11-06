using Distances: Chebyshev
using Neighborhood: bulkisearch, NeighborNumber
using SpecialFunctions: digamma

export Kraskov2

"""
    Kraskov2 <: MutualInformation
    Kraskov2(; k::Int = 1, metric_x = Chebyshev(), metric_y = Chebyshev(), base = ℯ)

The ``I^{(2)}(x, y)`` nearest neighbor based mutual information estimator from
Kraskov et al. (2004), using `k` nearest neighbors. The distance metric for
the marginals ``x`` and ``y`` can be chosen separately, while the `Chebyshev` metric
is always used for the `z = (x, y)` joint space.
"""
struct Kraskov2{MX, MY, MZ, B} <: MutualInformation
    k::Int
    metric_x::MX
    metric_y::MY
    metric_z::MZ # always Chebyshev, otherwise estimator is not valid!
    base::B

    function Kraskov2(; k::Int = 1, metric_x::MX = Chebyshev(), metric_y::MY = Chebyshev(),
            base::B = ℯ) where {MX, MY, B}
        metric_z = Chebyshev()

        new{MX, MY, typeof(metric_z), B}(k, metric_x, metric_y, Chebyshev(), base)
    end
end

function mutualinfo(est::Kraskov2,
        x::Vector_or_Dataset{D1, T}, y::Vector_or_Dataset{D2, T}) where {D1, D2, T}
    @assert length(x) == length(y)
    (; k, metric_x, metric_y, metric_z, base) = est

    z = Dataset(x, y)
    N = length(z)

    # Common for both kraskov estimators
    tree_z = KDTree(z, metric_z)
    idxs_z = bulkisearch(tree_z, z.data, NeighborNumber(k + 1))

    kth_nns_z = [idx_z[k + 1] for idx_z in idxs_z]
    ϵs_x = zeros(Float64, N)
    ϵs_y = zeros(Float64, N)
    eval_dists_to_knns!(ϵs_x, x, kth_nns_z, metric_x)
    eval_dists_to_knns!(ϵs_y, y, kth_nns_z, metric_y)

    # if the following equality holds for all points, then things are correct until this point
    #ϵ_maxes = [max(a, b) for (a, b) in zip(ϵs_x, ϵs_y)]
    #@assert all(ϵ_maxes .== ϵs_z)

    nx = zeros(Int, N)
    ny = zeros(Int, N)
    count_within_radius!(nx, x, metric_x, ϵs_x, N)
    count_within_radius!(ny, y, metric_y, ϵs_y, N)

    MI = digamma(est.k) -
        1/k -
        sum(digamma.(nx) .+ digamma.(ny)) / N +
        digamma(N)

    # Kraskov uses the natural logarithm (nats) in their derivation. Convert to target unit.
    return MI / log(base, ℯ)
end
