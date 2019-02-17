__precompile__(true)

module TransferEntropy

using Reexport

using Distributions
using Distances
using SpecialFunctions
using NearestNeighbors
using CausalityToolsBase
using StatsBase

include("TEVars.jl")

include("entropy.jl")

include("estimators/transferentropy_kraskov.jl")
include("estimators/transferentropy_visitfreq.jl")
include("estimators/transferentropy_transferoperator.jl")
include("estimators/common_interface.jl")
#include("estimators/transferentropy_triangulation.jl")


export
install_dependencies,
entropy,
TEVars
#transferentropy_transferoperator_triang, tetotri

end # module
