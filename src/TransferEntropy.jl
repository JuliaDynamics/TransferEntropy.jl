
module TransferEntropy


using Reexport

using Distributions
using Distances
using SpecialFunctions
using NearestNeighbors
@reexport using CausalityToolsBase
@reexport using PerronFrobenius
using StateSpaceReconstruction; export invariantize

import StatsBase
import DelayEmbeddings: AbstractDataset

include("EmbeddingTE.jl")
include("TEVars.jl")
include("Estimators/Estimators.jl")
include("interface.jl")

export
    transferentropy,
    TEVars,
    EmbeddingTE

#include("old_code/te_embed.jl")
#include("old_code/convenience_funcs_regular_te.jl")
#include("old_code/convenience_funcs_conditional_te.jl")

end # module
