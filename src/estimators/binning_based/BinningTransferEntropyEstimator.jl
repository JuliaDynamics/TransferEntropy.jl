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
abstract type BinningTransferEntropyEstimator end 

# Binning heuristics
include("binning_heuristics.jl")

# Binning-based estimators
include("TransferOperatorGrid.jl")
include("VisitationFrequency.jl")