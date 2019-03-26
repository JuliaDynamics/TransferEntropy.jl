import PerronFrobenius: 
	AbstractTriangulationInvariantMeasure

import CausalityToolsBase: 
	RectangularBinning
	
import StateSpaceReconstruction: 
	Simplex, 
	generate_interior_points

import StaticArrays: SVector

"""
transferentropy(μ::AbstractTriangulationInvariantMeasure, 
	binning_scheme::RectangularBinning, vars::TEVars; n::Int = 10000)

#### Transfer entropy using a precomputed invariant measure over a triangulated partition

Estimate transfer entropy from an invariant measure over a triangulation
that has been precomputed either as 

1. `μ = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())`, or
2. `μ = invariantmeasure(pts, TriangulationBinning(), ExactIntersection())` 

where the first method uses approximate simplex intersections (faster) and 
the second method uses exact simplex intersections (slow). `μ` contains 
all the information needed to compute transfer entropy. 

Note: `pts` must be a vector of states, not a vector of 
variables/(time series). Wrap your time series in a `Dataset`
first if the latter is the case.

#### Computing transfer entropy (triangulation -> rectangular partition)

Because we need to compute marginals, we need a rectangular grid. To do so,
transfer entropy is computed by sampling the simplices of the 
triangulation according to their measure with a total of approximately 
`n` points. Introducing multiple points as representatives for the partition
elements does not introduce any bias, because we in computing the 
invariant measure, we use no more information than what is encoded in the 
dynamics of the original data points. However, from the invariant measure,
we can get a practically infinite amount of points to estimate transfer 
entropy from.

Then, transfer entropy is estimated using the visitation 
frequency estimator on those points (see docs for `transferentropy_visitfreq` 
for more information), on a rectangular grid specified by `binning_scheme`.

#### Common use case

This method is good to use if you want to explore the sensitivity 
of transfer entropy to the bin size in the final rectangular grid, 
when you have few observations in the time series. The invariant 
measure, which encodes the dynamical information, is slow to compute over 
the triangulation, but only needs to be computed once.
After that, transfer entropy may be estimated at multiple scales very quickly.

### Example 

```julia
# Compute invariant measure over a triangulation using approximate 
# simplex intersections. This is relatively slow.
μ = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())

# Compute transfer entropy from the invariant measure over multiple 
# bin sizes. This is fast, because the measure has been precomputed.
tes = map(ϵ -> transferentropy(μ, RectangularBinning(ϵ), TEVars([1], [2], [3])), 2:50)
```
"""
function transferentropy(μ::AbstractTriangulationInvariantMeasure, 
		binning_scheme::RectangularBinning, vars::TEVars; n::Int = 10000)
	dim = length(μ.points[1])
	triang = μ.triangulation.simplexindices
	n_simplices = length(triang)

	simplices = [Simplex(μ.points[triang[i]]) for i = 1:n_simplices]

	# Find a number of points to fill each simplex with so that we 
	# obey the measure over the simplices of the triangulation.
	# The total number of points will be roughly `n`, but slightly 
	# higher because we need integer numbers of points and use `ceil`
	# for this.
	n_fillpts_persimplex = ceil.(Int, μ.measure.dist .* n)

	# Array to store the points filling the simplices
	fillpts = Vector{SVector{dim, Float64}}(undef, length(n_fillpts_persimplex))

	for i = 1:n_simplices
		sᵢ = simplices[i]
		if n_fillpts_persimplex[i] > 0
			pts = generate_interior_points(sᵢ, n_fillpts_persimplex[i])
			append!(fillpts, [SVector{dim, Float64}(pt) for pt in pts])
		end
	end

	transferentropy(fillpts, binning_scheme, vars, VisitationFrequency())
end