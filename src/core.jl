import Entropies:
    genentropy, ProbabilitiesEstimator, EntropyEstimator,
    VisitationFrequency, RectangularBinning,
    NaiveKernel, DirectDistance, TreeDistance,
    NearestNeighborEntropyEstimator, KozachenkoLeonenko, Kraskov

export VisitationFrequency, RectangularBinning, 
    NaiveKernel, DirectDistance, TreeDistance,
    KozachenkoLeonenko, Kraskov

import DelayEmbeddings: AbstractDataset, Dataset
export Dataset

using NearestNeighbors, Distances, SpecialFunctions



const Vector_or_Dataset{D, T} = Union{AbstractVector{T}, AbstractDataset{D, T}} where {D, T}
const Est = Union{ProbabilitiesEstimator, EntropyEstimator}
