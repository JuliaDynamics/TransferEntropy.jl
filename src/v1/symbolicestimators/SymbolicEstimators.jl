using Reexport

@reexport module SymbolicEstimators 
    include("SymbolicTransferEntropyEstimator.jl")
    include("SymbolicPerm.jl")
    include("SymbolicAmplitudeAware.jl")
    include("symbolic.jl")
    println("hey")
end

