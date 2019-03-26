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

E = cembed(hcat(pts...,))

te_grid_ll = transferentropy_transferoperator_grid(E, 0.3, vars)
te_grid_hl = transferentropy(pts, RectangularBinning(0.3), vars, TransferOperatorGrid())

te_vf_ll = transferentropy_visitfreq(E, 0.3, vars)
te_vf_hl = transferentropy(pts, RectangularBinning(0.3), vars, VisitationFrequency())

@show te_grid_ll, te_grid_hl
@show te_vf_ll, te_vf_hl

# We don't expect the estimates to be identical because the initial distribution over
# which the transfer operator is computed is 
@test te_grid_ll >= 0 
@test te_grid_hl >= 0

# The visitation frequency estimates must, however, be identical, because there is no
# randomisation involved in the estimation of the TE.
@test te_vf_ll >= 0 
@test te_vf_hl >= 0
@test te_vf_ll ≈ te_vf_hl
