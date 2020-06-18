include("interiorpts.jl")

export SimplexEstimator, transferentropy

import CausalityToolsBase: RectangularBinning
import PerronFrobenius
import ..TransferEntropyEstimator
import ..TEVars 
import ..BinningTransferEntropyEstimator
import .._transferentropy
import ..transferentropy
import ..EmbeddingTE 
import ..te_embed
import ..GridEstimators.BinningHeuristic
import ..estimate_partition

"""
    SimplexEstimator(to::M, te::E, n::Int = 5000, randomsampling::Bool = false) 
        where {M <: PerronFrobenius.TriangulationBasedTransferOperator, 
               E <: BinningTransferEntropyEstimator} → SimplexEstimator

A transfer entropy estimator that uses a combination of a triangulation based transfer 
operator estimator `to` of type `M <: TriangulationBasedTransferOperator` from 
PerronFrobenius.jl, and a transfer entropy estimator `te` from TransferEntropy.jl. 

Transfer entropy is estimated as follows: 
1. Triangulate the state space into disjoint simplices.
2. Approximate the transfer operator over elements of the triangulation.
3. From the transfer operator, compute an invariant measure over the triangulation elements. 
4. Generate a minimum of `n` points in the state space according to the invariant measure.
5. Estimate transfer entropy from those points using any non-triangulation-based transfer entropy estimator.

If `randomsampling == true`, then simplices of the triangulation are subsampled randomly within their 
interiors. If `randomsampling == false`, then simplices are subsampled according to a 
shape-preserving subvidivision of the simplices (if so, `n` is the minimum number of points,
because the number of subsampled points is dictated by the dimension and splitting factor).

*Note:* The use of this estimator requires loading the `Simplices.jl` package *after* loading TransferEntropy.jl. 

## Example

```julia
using TransferEntropy, Simplices
est_simplex_point = SimplexEstimator(SimplexPoint(), VisitationFrequency())
```
"""
struct SimplexEstimator{M <: PerronFrobenius.TriangulationBasedTransferOperator, E <: BinningTransferEntropyEstimator} <: TriangulationEstimator
    to::M
    te::E
    n::Int
    randomsampling::Bool
end
SimplexEstimator(to, te, n::Int = 5000, randomsampling::Bool = false) = SimplexEstimator(to, te, n, randomsampling)

Base.show(io::IO, method::SimplexEstimator) = print(io, "SimplexEstimator{$(method.to), $(method.te), $(method.n), $(method.randomsampling)}")


const GEN{TO} = PerronFrobenius.TransferOperatorGenerator{TO} where TO <: PerronFrobenius.TriangulationBasedTransferOperator
const TO = PerronFrobenius.TransferOperatorApproximation

""" Generate points according to the given invariant distribution over a triangulation."""
function generate_μpts(μ::PerronFrobenius.InvariantDistribution{<:TO{<:GEN}}, n::Int, randomsampling::Bool)
    pts = μ.to.g.init.invariant_pts
    triang = μ.to.g.init.triang
    
    n_persimplex = ceil(Int, n/length(triang))
    μpts = Vector{SVector}(undef, 0)

    for simplex in triang
        simplex_vertices = pts[simplex]
        append!(μpts, interiorpts(simplex_vertices, n_persimplex, randomsampling))
    end
    
    return μpts
end


# Compute transfer entropy over points generated from an invariant distribution over a triangulation 
# using a binning-based transfer entropy estimator.
function transferentropy(μ::PerronFrobenius.InvariantDistribution{<:TO{<:GEN}}, vars::TEVars, 
        binning::RectangularBinning, estimator::BinningTransferEntropyEstimator, 
        randomsampling::Bool = false, n::Int = 300)

    _transferentropy(generate_μpts(μ, n, randomsampling), vars, binning, estimator)
end


function transferentropy(μ::PerronFrobenius.InvariantDistribution{<:TO{<:GEN}}, vars::TEVars, 
        estimator::BinningTransferEntropyEstimator, 
        randomsampling::Bool = false, n::Int = 10000)

    # Get the binning (if a heuristic is used, determine binning from input time series and dimension)
    if estimator.binning isa BinningHeuristic
        dim = length(μ.to.g.pts.data[1])
        binning = estimate_partition(n, dim, estimator.binning)
    else
        binning = estimator.binning
    end

    # Generate the points 
    pts = generate_μpts(μ, n, randomsampling)

    # Compute TE over different partitions and summarize
    bs = binning isa Vector{RectangularBinning} ? binning : [binning]
    tes = map(binscheme -> _transferentropy(pts, vars, binscheme, estimator), bs)
    te = estimator.summary_statistic(tes)

    return te
end


function transferentropy(source, target, embedding::EmbeddingTE, method::SimplexEstimator{TO, TE}) where {TO, TE}
    
    # Generalized delay embedding
    pts, vars, τs, js = te_embed(source, target, embedding)
    
    # Compute invariant measure
    μ = PerronFrobenius.invariantmeasure(pts, method.to)
    
    # Transfer entropy over points generated from that invariant measure using 
    # the provided transfer entropy estimator (method.te).
    transferentropy(μ, vars, method.te, method.randomsampling, method.n)
end

function transferentropy(source, target, cond, embedding::EmbeddingTE, method::SimplexEstimator{TO, TE}) where {TO, TE}
    
    # Generalized delay embedding
    pts, vars, τs, js = te_embed(source, target, cond, embedding)
    
    # Compute invariant measure
    μ = PerronFrobenius.invariantmeasure(pts, method.to)
    
    # Transfer entropy over points generated from that invariant measure using 
    # the provided transfer entropy estimator (method.te).
    transferentropy(μ, vars, method.te, method.randomsampling, method.n)
end