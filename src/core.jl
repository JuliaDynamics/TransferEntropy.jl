using Reexport
@reexport using Entropies

import DelayEmbeddings: AbstractDataset, Dataset
export Dataset

using Distances, SpecialFunctions

const Vector_or_Dataset{D, T} = Union{AbstractVector{T}, AbstractDataset{D, T}} where {D, T}
const Est = Union{ProbabilitiesEstimator}
