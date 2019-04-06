# Release v0.4.2

## New functionality

- Added `te_embed` function that for regular and conditional transfer entropy analyses constructs appropriate delay reconstruction points and corresponding `TEVars` instances that instructs the `transferentropy` methods how to compute the marginals.

# Release v0.4.1

- Update documentation with more examples.
- Fix bug in triangulation estimator where multiple points at the origin was used to sample 
        simplices. This bug only affected the new methods with updates syntax from v`0.4.0`.

# Release v0.4.0

New syntax for the different estimators.

## Rectangular binnings

- `transferentropy(pts, v::TEVars, binning_scheme::RectangularBinning, VisitationFrequency())` uses a regular visitation frequency estimator.

- `transferentropy(pts, v::TEVars, binning_scheme::RectangularBinning, TransferOperatorGrid())` uses the transfer operator grid estimator.

## Triangulation binnings

For computing transfer entropy from triangulations, first compute the invariant measure 
over the triangulation, then superimpose a rectangular grid and compute the transfer 
entropy over that grid. For a precomputed invariant meausre, the syntax is: 

- `transferentropy(μ::AbstractTriangulationInvariantMeasure, vars::TEVars,
        binning_scheme::RectangularBinning; n::Int = 10000)`.

For example:

```julia
μapprox = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())
μexact = invariantmeasure(pts, TriangulationBinning(), ExactIntersection())

# Compute transfer entropy at single bin size
transferentropy(μapprox, vars, RectangularBinning(0.2))
transferentropy(μexact, vars, RectangularBinning(0.2))

# Compute transfer entropy from the invariant measure over multiple 
# bin sizes. This is fast, because the measure has been precomputed.
binsizes = [0.2, 0.3, 0.5, 0.7]
[transferentropy(μapprox, vars, RectangularBinning(bs)) for bs in binsizes]
[transferentropy(μexact, vars, RectangularBinning(bs)) for bs in binsizes]
```