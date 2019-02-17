using CausalityToolsBase
using StateSpaceReconstruction

pts = invariantize([rand(3) for i = 1:15])
spts = [SVector{3, Float64}(pt) for pt in pts]
mpts = [MVector{3, Float64}(pt) for pt in pts]
D = Dataset(pts)

vars = TEVars([1], [2], [3])
@test transferentropy(pts, RectangularBinning(0.2), vars) >= 0
@test transferentropy(spts, RectangularBinning(0.2), vars) >= 0
@test transferentropy(mpts, RectangularBinning(0.2), vars) >= 0
@test transferentropy(D, RectangularBinning(0.2), vars) >= 0


# Compute invariant measure over a triangulation using approximate 
# simplex intersections. This is relatively slow.
μapprox = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())
μexact = invariantmeasure(pts, TriangulationBinning(), ExactIntersection())

# Compute transfer entropy from the invariant measure over multiple 
# bin sizes. This is fast, because the measure has been precomputed.
@test transferentropy(μapprox, RectangularBinning(0.2), vars) >= 0
@test transferentropy(μexact, RectangularBinning(0.2), vars) >= 0

@test transferentropy_transferoperator_grid(E, 0.3, vars) .≈ transferentropy(pts, RectangularBinning(0.3), vars, TransferOperatorGrid())
@test transferentropy_visitfreq(E, 0.3, vars) .≈ transferentropy(pts, RectangularBinning(0.3), vars, VisitationFrequency())
