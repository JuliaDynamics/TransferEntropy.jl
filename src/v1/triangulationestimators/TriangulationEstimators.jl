using Reexport

@reexport module SimplexEstimators
    include("TriangulationEstimator.jl")
    include("SimplexEstimator.jl")
end # module