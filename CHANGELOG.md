# Changelog

## Release v1.1.0

- Internal API changes. Not affecting end user.
- Documentation update.

## Release v1.0.0

- Completely new API. See documentation for details.

## Release v0.4.3

### New functionality

Added convenience methods to compute regular transfer entropy directly from time series. The following methods work:

1. `transferentropy(source, driver; dim = 3, τ = 1, η = 1)`: compute transfer entropy from `source` to `target` using a `dim`-dimensional delay reconstruction with prediction lag `η` and embedding lag `τ`, inferring appropriate bin sizes from time series length. Extra dimensions are assigned to the ``T_{pp}`` component of the delay reconstruction. Returns the result over those binsizes. 

2. `transferentropy(source, driver, Union{::RectangularBinning, AbstractArray{::RectangularBinning}}, dim = 3, τ = 1, η = 1)`: compute transfer entropy from `source` to `target` using a `dim`-dimensional delay reconstruction with prediction lag `η` and embedding lag `τ`, specifying the partition(s) explicitly. Extra dimensions are assigned to the ``T_{pp}`` component of the delay reconstruction. Returns one transfer entropy estimate per partition. 

3. `transferentropy(source, driver, k::Int, l::Int, m::Int; τ = 1, η = 1)`: compute transfer entropy from `source` to `target` using a `dim`-dimensional delay reconstruction with `k` being the dimension of the ``T_f`` component, `l` the dimension of the ``T_{pp}`` component and `m` the dimension of the ``S_{pp}`` component, using prediction lag `η` and embedding lag `τ`, inferring appropriate bin sizes from time series length. Returns the result over those binsizes. 

4. `transferentropy(source, driver, k::Int, l::Int, m::Int; τ = 1, η = 1)`: compute transfer entropy from `source` to `target` using a `dim`-dimensional delay reconstruction with `k` being the dimension of the ``T_f`` component, `l` the dimension of the ``T_{pp}`` component and `m` the dimension of the ``S_{pp}`` component, using prediction lag `η` and embedding lag `τ`, specifying the partition(s) explicitly. Returns one transfer entropy estimate per partition. 

The same works for computing conditional transfer entropy with three time series, but an extra time series and embedding component must be added, i.e. `transferentropy(source, driver, cond; dim = 4, τ = 1, η = 1)` or `transferentropy(source, driver, cond; k::Int, l::Int, m::Int, n::Int; τ = 1, η = 1`, where `m` is the dimension of the ``C_{pp}`` component of the delay reconstruction. 

An optional `estimator` keyword may be provided. The default is `estimator = VisitationFrequency()`, but `estimator = TransferOperatorGrid()` also works. For triangulation-based estimators, you still need to compute the invariant measure manually first, then compute transfer entropy as usual.

## Release v0.4.2

### New functionality

- Added `te_embed` function that for regular and conditional transfer entropy analyses constructs appropriate delay reconstruction points and corresponding `TEVars` instances that instructs the `transferentropy` methods how to compute the marginals.

## Release v0.4.1

- Update documentation with more examples.
- Fix bug in triangulation estimator where multiple points at the origin was used to sample 
        simplices. This bug only affected the new methods with updates syntax from v`0.4.0`.

## Release v0.4.0

New syntax for the different estimators.

### Rectangular binnings

- `transferentropy(pts, v::TEVars, binning_scheme::RectangularBinning, VisitationFrequency())` uses a regular visitation frequency estimator.

- `transferentropy(pts, v::TEVars, binning_scheme::RectangularBinning, TransferOperatorGrid())` uses the transfer operator grid estimator.

### Triangulation binnings

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
