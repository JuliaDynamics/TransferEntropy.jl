module TransferEntropy
using Reexport
@reexport using Entropies
include("core.jl")
include("mutualinfo/interface.jl")
include("transferentropy/interface.jl")
end # module
