import PerronFrobenius: AbstractTriangulationInvariantMeasure
import CausalityToolsBase: RectangularBinning, CustomReconstruction
import StateSpaceReconstruction: Simplex, generate_interior_points
import StaticArrays: SVector
import Distances: Metric, Chebyshev
import StatsBase 

export transferentropy, 
    TransferEntropyEstimator, 
        TransferOperatorGrid, 
        VisitationFrequency, 
        NearestNeighbourMI

"""
    TransferEntropyEstimator

An abstract type for transfer entropy estimators. This type has several concrete subtypes
that are accepted as inputs to the [`transferentropy`](@ref) methods. 

- [`VisitationFrequency`](@ref)
- [`TransferOperatorGrid`](@ref)
- [`NearestNeighbourMI`](@ref)
"""
abstract type TransferEntropyEstimator end 

function Base.show(io::IO, estimator::TransferEntropyEstimator)
    s = "$(typeof(estimator))($(estimator.b))"
    print(io, s)
end

include("binning_based/TransferOperatorGrid.jl")
include("binning_based/VisitationFrequency.jl")
include("nearest_neighbour_based/NearestNeighbourMI.jl")

# Low-level implementations (on point clouds with TEVars instances)
include("estimators/old_code/transferentropy_kraskov.jl")
include("estimators/old_code/transferentropy_visitfreq.jl")
include("estimators/old_code/transferentropy_transferoperator.jl")
include("estimators/old_code/common_interface.jl")
include("estimators/old_code/nearestneighbourMI_estimator.jl")