
module TransferEntropy

import CausalityToolsBase: 
    RectangularBinning

export RectangularBinning

import PerronFrobenius: 
    SingleGrid

export SingleGrid

include("v1/TEVars.jl")
include("v1/EmbeddingTE.jl")
include("v1/api.jl")

include("v1/gridestimators/GridEstimators.jl")
include("v1/neighborestimators/NearestNeighborEstimators.jl")
include("v1/symbolicestimators/SymbolicEstimators.jl")

using Requires 
function __init__()
    @require Simplices="d5428e67-3037-59ba-9ab1-57a04f0a3b6a" begin
        #@require PerronFrobenius="260eed61-d0e8-5f1e-b040-a9756a401c97" begin

            import PerronFrobenius: SimplexExact, SimplexPoint
            export SimplexExact, SimplexPoint
        
            include("v1/triangulationestimators/TriangulationEstimators.jl")
        #end
    end
end
# using Reexport

# using Distributions
# using Distances
# using SpecialFunctions
# using NearestNeighbors
# @reexport using CausalityToolsBase
# @reexport using PerronFrobenius
# using StateSpaceReconstruction; export invariantize

# import StatsBase
# import DelayEmbeddings: AbstractDataset

# include("EmbeddingTE.jl")
# include("TEVars.jl")
# include("Estimators/Estimators.jl")
# include("interface.jl")

# export
#     transferentropy,
#     TEVars,
#     EmbeddingTE

#include("old_code/te_embed.jl")
#include("old_code/convenience_funcs_regular_te.jl")
#include("old_code/convenience_funcs_conditional_te.jl")

end # module
