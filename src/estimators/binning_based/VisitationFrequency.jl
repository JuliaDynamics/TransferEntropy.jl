export VisitationFrequency

"""
    VisitationFrequency(; b::Number = 2)

An transfer entropy estimator which computes transfer entropy over a 
dicretization of an appropriate [delay reconstruction](@ref custom_delay_reconstruction) from the 
input time series [^1]. The invariant probabilities over the partition are estimated 
using a simple counting approach.

## Fields 

- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).
- **`summary_statistic::Function = StatsBase.mean`**: The summary statistic to use if multiple discretization schemes are given.
- **`binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic}`**: 
    The discretization scheme. Can either be fixed (i.e. one or more `RectangularBinning` instances),
    or a `BinningHeuristic`. In the latter case, the binning is determined from the input data.

## References

[^1]:
    Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation 
    using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
Base.@kwdef struct VisitationFrequency <: BinningTransferEntropyEstimator
    """ The base of the logarithm usen when computing transfer entropy. """
    b::Number = 2.0

    """ The summary statistic to use if multiple discretization schemes are given """
    summary_statistic::Function = StatsBase.mean

    """ The discretization scheme. """
    binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic} = ExtendedPalusLimit()

    VisitationFrequency(b, summary_statistic, binning) = new(b, summary_statistic, binning)
end