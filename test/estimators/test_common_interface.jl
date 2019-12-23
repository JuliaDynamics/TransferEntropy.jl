using CausalityToolsBase
using StateSpaceReconstruction

# You'd use an 
pts = invariantize([rand(3) for i = 1:200])
spts = [SVector{3, Float64}(pt) for pt in pts]
mpts = [MVector{3, Float64}(pt) for pt in pts]
D = Dataset(pts)

# Initialising test
@test VisitationFrequency().b isa Number
@test VisitationFrequency(b = 2).b isa Number
@test VisitationFrequency(b = 10).b isa Number
@test TransferOperatorGrid().b isa Number
@test TransferOperatorGrid(b = 10).b isa Number

vars = TEVars([1], [2], [3])
@test transferentropy(pts, vars, RectangularBinning(0.2), VisitationFrequency()) >= 0
@test transferentropy(spts, vars, RectangularBinning(0.2), VisitationFrequency()) >= 0
@test transferentropy(mpts, vars,  RectangularBinning(0.2), VisitationFrequency()) >= 0
@test transferentropy(D, vars, RectangularBinning(0.2), VisitationFrequency()) >= 0

# Different logarithms should give different TE values
te_b2 = transferentropy(pts, vars, RectangularBinning(0.2), VisitationFrequency(b = 2))
te_b10 = transferentropy(pts, vars, RectangularBinning(0.2), VisitationFrequency(b = 10))
@test te_b2 != te_b10

@test transferentropy(pts, vars, RectangularBinning(0.2), TransferOperatorGrid()) >= 0
@test transferentropy(spts, vars, RectangularBinning(0.2), TransferOperatorGrid()) >= 0
@test transferentropy(mpts, vars, RectangularBinning(0.2), TransferOperatorGrid()) >= 0
@test transferentropy(D, vars, RectangularBinning(0.2), TransferOperatorGrid()) >= 0


@test NearestNeighbourMI(b = 10).b isa Number


pts = invariantize([rand(3) for i = 1:15])
spts = [SVector{3, Float64}(pt) for pt in pts]
mpts = [MVector{3, Float64}(pt) for pt in pts]
D = Dataset(pts)

te_D = transferentropy(D, vars, NearestNeighbourMI(b = 2))
te_pts = transferentropy(pts, vars, NearestNeighbourMI(b = 2))
te_spts = transferentropy(spts, vars, NearestNeighbourMI(b = 2))
te_mpts = transferentropy(mpts, vars, NearestNeighbourMI(b = 2))

@test te_D isa Number
@test te_pts isa Number
@test te_spts isa Number
@test te_mpts isa Number

te_D ≈ te_pts ≈ te_spts ≈ te_mpts



# Compute invariant measure over a triangulation using approximate and exact
# simplex intersections. This is relatively slow.
μapprox = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())
μexact = invariantmeasure(pts, TriangulationBinning(), ExactIntersection())

# Compute transfer entropy from the invariant measure over multiple 
# bin sizes. This is fast, because the measure has been precomputed.
@test transferentropy(μapprox, vars, RectangularBinning(0.2)) >= 0
@test transferentropy(μexact, vars, RectangularBinning(0.2)) >= 0

#@test transferentropy_transferoperator_grid(pts, 0.3, vars), transferentropy(pts,  vars, RectangularBinning(0.3),TransferOperatorGrid())
#@test transferentropy_visitfreq(pts, 0.3, vars), transferentropy(pts, vars, RectangularBinning(0.3), VisitationFrequency())
