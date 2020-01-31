export TransferOperatorGrid

"""
    TransferOperatorGrid(; b::Number = 2, summary_statistic = StatsBase.mean, 
        binning = ExtendedPalusLimit())

An transfer entropy estimator which computes transfer entropy over a 
dicretization of an appropriate [delay reconstruction](@ref custom_delay_reconstruction) of the input data.
Invariant probabilities over the partition are computed using an approximation to the transfer (Perron-Frobenius) 
operator over the grid [1], which explicitly gives the transition probabilities between states. 
The transfer entropy is computed using the logarithm to base `b`. 

## Fields 

- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).
- **`summary_statistic::Function = StatsBase.mean`**: The summary statistic to use if multiple discretization schemes are given.
- **`binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic} = ExtendedPalusLimit()`**: 
    The discretization scheme. Can either be fixed (i.e. one or more `RectangularBinning` instances),
    or a `BinningHeuristic`. In the latter case, the binning is determined from the input data.

## References

[^1]:
    Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation 
    using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
Base.@kwdef struct TransferOperatorGrid <: BinningTransferEntropyEstimator
    """ The base of the logarithm usen when computing transfer entropy. """
    b::Number = 2.0

    """ The summary statistic to use if multiple discretization schemes are given """
    summary_statistic::Function = StatsBase.mean

    """ The discretization scheme. """
    binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic} = ExtendedPalusLimit()

    TransferOperatorGrid(b, summary_statistic, binning) = new(b, summary_statistic, binning)
end