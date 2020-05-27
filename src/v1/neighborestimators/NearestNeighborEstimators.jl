using Reexport 

@reexport module NearestNeighbourEstimators 
    # Supertype for all nearest neighbor-type estimators
    include("NearestNeighborTransferEntropyEstimator.jl")

    # Computing distances and nearest neighbors between statically sized vectors
    include("distances_tree.jl")

    # Estimators
    include("NearestNeighborMI.jl")
end