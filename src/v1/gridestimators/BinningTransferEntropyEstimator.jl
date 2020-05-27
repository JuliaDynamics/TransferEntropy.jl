export transferentropy, BinningTransferEntropyEstimator

import ..EmbeddingTE
import ..te_embed
import ..transferentropy
"""
    BinningTransferEntropyEstimator <: TransferEntropyEstimator

An abstract type for transfer entropy estimators that works on a discretization 
of the [reconstructed state space](@ref custom_delay_reconstruction). Has the following concrete subtypes

- [`VisitationFrequency`](@ref)
- [`TransferOperatorGrid`](@ref)

## Used by

Concrete subtypes are accepted as inputs by

- [`transferentropy`](@ref te_estimator_rectangular) (low-level method)
"""
abstract type BinningTransferEntropyEstimator <: TransferEntropyEstimator end 

import CausalityToolsBase: BinningHeuristic, RectangularBinning

function transferentropy(source, target, embedding::EmbeddingTE, estimator::BinningTransferEntropyEstimator)
    # Generalised delay embedding
    pts, vars, lags = te_embed(source, target, embedding)

    # Get the binning (if a heuristic is used, determine binning from input time series and dimension)
    if estimator.binning isa BinningHeuristic
        total_dim = length(pts[1])
        binning = GridEstimators.estimate_partition(target, total_dim, estimator.binning)
    else
        binning = estimator.binning
    end

    # Compute TE over different partitions
    bs = binning isa Vector{RectangularBinning} ? binning : [binning]
    tes = map(binscheme -> _transferentropy(pts, vars, binscheme, estimator), bs)

    return estimator.summary_statistic(tes)
end

function transferentropy(source, target, cond, embedding::EmbeddingTE, estimator::BinningTransferEntropyEstimator)
    # Generalised delay embedding
    pts, vars, lags = te_embed(source, target, cond, embedding)

    # Get the binning (if a heuristic is used, determine binning from input time series and dimension)
    if estimator.binning isa BinningHeuristic
        total_dim = length(pts[1])
        binning = GridEstimators.estimate_partition(target, total_dim, estimator.binning)
    else
        binning = estimator.binning
    end

    # Compute TE over different partitions
    bs = binning isa Vector{RectangularBinning} ? binning : [binning]
    tes = map(binscheme -> _transferentropy(pts, vars, binscheme, estimator), bs)

    return estimator.summary_statistic(tes)
end