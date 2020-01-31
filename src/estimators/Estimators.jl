using Reexport 

@reexport module Estimators

    import CausalityToolsBase: RectangularBinning
    import Distances: Metric, Chebyshev
    using StatsBase

    include("TransferEntropyEstimator.jl")

    # Binning based estimators
    # ----------------------------------
    include("binning_based/BinningTransferEntropyEstimator.jl")

    # Nearest neighbour based estimators
    # ----------------------------------
    include("nearest_neighbour_based/NearestNeighbourTransferEntropyEstimator.jl")

    export TransferEntropyEstimator,
        NearestNeighbourMI,
        VisitationFrequency,
        TransferOperatorGrid,
        # Binning heuristics
        PalusLimit,
        ExtendedPalusLimit
end