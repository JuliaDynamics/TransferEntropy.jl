using Reexport
@reexport using Entropies

import DelayEmbeddings: AbstractDataset, Dataset
export Dataset

using Distances, SpecialFunctions

const Vector_or_Dataset{D, T} = Union{AbstractVector{T}, AbstractDataset{D, T}} where {D, T}
const Est = Union{ProbabilitiesEstimator}


abstract type InformationCausalityMeasure end

function estimate_from_marginals(measure::I, args...;
        kwargs...) where I <: InformationCausalityMeasure
    msg = "Marginal estimation of $I is not implemented"
    throw(ArgumentError(msg))
end
