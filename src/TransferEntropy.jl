__precompile__(true)

module TransferEntropy

using Reexport

using Distributions
using Distances
using SpecialFunctions
using NearestNeighbors
using CausalityToolsBase
using StatsBase

export 
TransferEntropyEstimator, 
TransferOperatorGrid, 
VisitationFrequency, 
NNEstimator,
TEVars,
transferentropy

abstract type TransferEntropyEstimator end 

struct TransferOperatorGrid <: TransferEntropyEstimator end
struct VisitationFrequency <: TransferEntropyEstimator end
struct NNEstimator <: TransferEntropyEstimator end


include("TEVars.jl")

# keep old estimators for backwards compatibility
include("compat/compat_transferentropy_kraskov.jl")
include("compat/compat_transferentropy_visitfreq.jl")
include("compat/compat_transferentropy_transferoperator.jl")

# New interface
include("estimators/transferentropy_visitfreq.jl")
include("estimators/transferentropy_transferoperator_grid.jl")
include("estimators/transferentropy_transferoperator_triang.jl")
include("estimators/transferentropy_nearestneighbour.jl")

include("estimators/common_interface.jl")

end # module
