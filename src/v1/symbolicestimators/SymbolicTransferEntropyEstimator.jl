export SymbolicTransferEntropyEstimator

import StatsBase 
import DelayEmbeddings: genembed, Dataset

import ..TransferEntropyEstimator

abstract type SymbolicTransferEntropyEstimator <: TransferEntropyEstimator end

"""
symbolize

Symbolize time series or embeddings according to some method.

symbolize(E::Dataset, method::SymbolicPerm)

Permutation symbolization of vectors `E`. The dimension `m` (symbol sequence 
length) is determined from the dimension of the embedding vectors. 
"""
function symbolize end