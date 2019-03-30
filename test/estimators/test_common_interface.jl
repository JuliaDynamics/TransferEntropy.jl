using CausalityToolsBase
using StateSpaceReconstruction

# You'd use an 
pts = invariantize([rand(3) for i = 1:200])
spts = [SVector{3, Float64}(pt) for pt in pts]
mpts = [MVector{3, Float64}(pt) for pt in pts]
D = Dataset(pts)

vars = TEVars([1], [2], [3])
@test transferentropy(pts, vars, RectangularBinning(0.2), VisitationFrequency()) >= 0
@test transferentropy(spts, vars, RectangularBinning(0.2), VisitationFrequency()) >= 0
@test transferentropy(mpts, vars,  RectangularBinning(0.2), VisitationFrequency()) >= 0
@test transferentropy(D, vars, RectangularBinning(0.2), VisitationFrequency()) >= 0


@test transferentropy(pts, vars, RectangularBinning(0.2), TransferOperatorGrid()) >= 0
@test transferentropy(spts, vars, RectangularBinning(0.2), TransferOperatorGrid()) >= 0
@test transferentropy(mpts, vars, RectangularBinning(0.2), TransferOperatorGrid()) >= 0
@test transferentropy(D, vars, RectangularBinning(0.2), TransferOperatorGrid()) >= 0



pts = invariantize([rand(3) for i = 1:15])
spts = [SVector{3, Float64}(pt) for pt in pts]
mpts = [MVector{3, Float64}(pt) for pt in pts]
D = Dataset(pts)


# Compute invariant measure over a triangulation using approximate 
# simplex intersections. This is relatively slow.
μapprox = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())
μexact = invariantmeasure(pts, TriangulationBinning(), ExactIntersection())

# Compute transfer entropy from the invariant measure over multiple 
# bin sizes. This is fast, because the measure has been precomputed.
@test transferentropy(μapprox, vars, RectangularBinning(0.2)) >= 0
@test transferentropy(μexact, vars, RectangularBinning(0.2)) >= 0

#@test transferentropy_transferoperator_grid(pts, 0.3, vars), transferentropy(pts,  vars, RectangularBinning(0.3),TransferOperatorGrid())
#@test transferentropy_visitfreq(pts, 0.3, vars), transferentropy(pts, vars, RectangularBinning(0.3), VisitationFrequency())
