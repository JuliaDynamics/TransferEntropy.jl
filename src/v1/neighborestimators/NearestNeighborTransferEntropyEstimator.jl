
export transferentropy, NearestNeighborTransferEntropyEstimator

import ..TransferEntropyEstimator
import ..te_embed 
import ..EmbeddingTE
import ..transferentropy

"""
    NearestNeighborTransferEntropyEstimator{M} <: TransferEntropyEstimator

The supertype of all nearest neighbor based transfer entropy estimators,
where `M` is the type of metric.

Has the following concrete subtypes:

- [`NearestNeighborMI`](@ref)
"""
abstract type NearestNeighborTransferEntropyEstimator{M} <: TransferEntropyEstimator end 

function transferentropy(source, target, embedding::EmbeddingTE, 
        estimator::NearestNeighborTransferEntropyEstimator{M}) where M
    # Generalised delay embedding
    pts, vars, τs, js = te_embed(source, target, embedding)
    
    _transferentropy(pts, vars, estimator)
end

function transferentropy(source, target, cond, embedding::EmbeddingTE, 
        estimator::NearestNeighborTransferEntropyEstimator{M}) where M
    # Generalised delay embedding
    pts, vars, τs, js = te_embed(source, target, cond, embedding)

    _transferentropy(pts, vars, estimator)
end