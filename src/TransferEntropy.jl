
module TransferEntropy

import CausalityToolsBase: RectangularBinning
export RectangularBinning

import PerronFrobenius: SingleGrid
export SingleGrid

include("TEVars.jl")
include("EmbeddingTE.jl")
include("api.jl")

include("gridestimators/GridEstimators.jl")
include("neighborestimators/NearestNeighborEstimators.jl")
include("symbolicestimators/SymbolicEstimators.jl")

using Requires 
function __init__()
    @require Simplices="d5428e67-3037-59ba-9ab1-57a04f0a3b6a" begin
        #@require PerronFrobenius="260eed61-d0e8-5f1e-b040-a9756a401c97" begin

            import PerronFrobenius: SimplexExact, SimplexPoint
            export SimplexExact, SimplexPoint
        
            include("triangulationestimators/TriangulationEstimators.jl")
        #end
    end
end

end # module
