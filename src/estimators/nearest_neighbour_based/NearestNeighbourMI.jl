"""
    NearestNeighbourMI(k1::Int = 2, k2::Int = 3, metric::Metric = Chebyshev, b::Number)

A transfer entropy estimator using counting of nearest neighbours nearest neighbours 
to estimate mutual information over an appropriate 
[custom delay reconstruction](@ref custom_delay_reconstruction) of the input data. 
(the method from Kraskov et al. (2004)[^1], as implemented in Diego et al. (2019)[^2]).

## Fields 

- **`k1::Int = 2`**: The number of nearest neighbours for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- **`k2::Int = 3`**: The number of nearest neighbours for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- **`metric::Metric = Chebyshev()`**: The metric used for distance computations.
- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).

## References

[^1]:
    Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
    mutual information." Physical review E 69.6 (2004): 066138.
[^2]:
    Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation 
    using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
@Base.kwdef struct NearestNeighbourMI <: NearestNeighbourTransferEntropyEstimator
    k1::Int = 2
    k2::Int = 3
    metric::Metric = Chebyshev()
    b::Number = 2
end

function Base.show(io::IO, estimator::NearestNeighbourMI)
    k1 = estimator.k1
    k2 = estimator.k2
    metric = estimator.metric
    b = estimator.b
    s = "$(typeof(estimator))(b=$(b), k1=$(k1), k2=$(k2), metric=$(typeof(metric)))"
    print(io, s)
end