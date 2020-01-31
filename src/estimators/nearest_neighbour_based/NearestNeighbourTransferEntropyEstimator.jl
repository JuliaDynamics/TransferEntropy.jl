"""
    NearestNeighbourTransferEntropyEstimator <: TransferEntropyEstimator

An abstract type for transfer entropy estimators that works on a discretization 
of the [reconstructed state space](@ref custom_delay_reconstruction). 

Has the following concrete subtypes:

- [`NearestNeighbourMI`](@ref)
"""
abstract type NearestNeighbourTransferEntropyEstimator <: TransferEntropyEstimator end 

include("NearestNeighbourMI.jl")