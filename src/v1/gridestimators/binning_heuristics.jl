import CausalityToolsBase: BinningHeuristic

export PalusLimit, ExtendedPalusLimit

"""
    PalusLimit(max_subdivs::Int = 8) <: BinningHeuristic

Partition the state space by by splitting each marginal into at most ``N^{1/(D+1)}``
equally-spaced intervals, where ``N`` is the number of observations and ``D`` is the 
dimension of the state space. This results in a rectangular binning.
"""
@Base.kwdef struct PalusLimit <: BinningHeuristic
    max_subdivs::Int = 8
end


function Base.show(io::IO, x::PalusLimit)
    print(io, "PalusLimit(max_subdivs = $(x.max_subdivs))")
end


"""
    ExtendedPalusLimit(ext_finer = 1, ext_coarser = 0, min_subdivs = 2, max_subdivs = 8) <: BinningHeuristic

The [`PalusLimit`](@ref) heuristic partitions the state space by splitting each marginal into at 
most ``N^{1/(D+1)}`` equally-spaced intervals, where ``N`` is the number of observations 
and ``D`` is the dimension of the state space. This results in a rectangular binning.

`ExtendedPalusLimit` differs from `PalusLimit` in that it gives the 
possibility of adding more than one partition in addition to that obtained 
by using `PalusLimit`. The extra partition(s) can be either finer 
(e.g. `ext_finer = 1` adds one more finer partition), or coarser
 (e.g. `ext_coarser = 1` adds one more coarser partitions). 
The minimum number of subdivisions is downwards limited by 
`min_subdivs`  (giving maximum coarseness of the partition) and upwards 
limited by `max_subdivs` (giving maximum fineness of the partition).
"""
@Base.kwdef struct ExtendedPalusLimit <: BinningHeuristic
    ext_finer::Int = 1
    ext_coarser::Int = 0
    min_subdivs::Int = 2
    max_subdivs::Int = 8
    
    function ExtendedPalusLimit(ext_finer, ext_coarser, min_subdivs, max_subdivs)
        if min_subdivs < 2
            throw(ArgumentError("min_subdivs needs to be at least 2 (otherwise the marginals will not be subdivided at all!)"))
        end
        new(ext_finer, ext_coarser, min_subdivs, max_subdivs)
    end
end

function Base.show(io::IO, x::ExtendedPalusLimit)
    print(io, "ExtendedPalusLimit(ext_finer=$(x.ext_finer), ext_coarser=$(x.ext_coarser), min_subdivs = $(x.min_subdivs), max_subdivs = $(x.max_subdivs))")
end

"""
    estimate_partition(x::AbstractVector{T}, dim::Int, heuristic::BinningHeuristic) -> RectangularBinning

Estimate a suitable partition for a `dim`-dimensional state space reconstruction of the data series `x` 
using the provided binning `heuristic`.

The heuristic can be any of the following:

- **[`PalusLimit(max_subdivs = 8)`](@ref)**: The partition will be constructed by splitting each of the `dim` marginals 
    into at most ``N^{\\frac{1}{D+1}}`` equally-spaced intervals each, resulting in a rectangular binning. The number 
    of intervals will be limited upwards by `max_subdivs` (e.g. giving the maximum fineness of the partition).
- **[`ExtendedPalusLimit(ext_finer = 0, ext_coarser = 0, max_subdivs = 8)`](@ref)**. The same as for `PalusLimit`, 
    but with the possibility of adding more than one partition in addition to that obtained by using `PalusLimit`.
    The extra partition(s) can be either finer (e.g. `ext_finer = 2` adds one more finer partition), or coarser
    (e.g. `ext_coarser = 1` adds one more coarser partitions). The extra partition(s) can be either finer 
    (e.g. `ext_finer = 2` adds one more finer partition), or coarser (e.g. `ext_coarser = 1` adds one more 
    coarser partitions). The minimum number of subdivisions is downwards limited by 
    `min_subdivs`  (giving maximum coarseness of the partition) and upwards limited by `max_subdivs`
    (giving maximum fineness of the partition).

## Examples 

### `PalusLimit`

Using default number of maximum subdivisions along each marginal: 

```jldoctest
using CausalityTools
x = rand(1000) # our time series
D = 3 # dimension of the desired state space reconstruction
estimate_partition(x, D, PalusLimit())

# output
RectangularBinning(5)
```

Limiting maximum number of intervals along each marginal to 5: 


```jldoctest
using CausalityTools
estimate_partition(rand(1000), 4, PalusLimit(max_subdivs = 5))

# output
RectangularBinning(3)
```

### `ExtendedPalusLimit`

Getting three distinct binning, extending from what is obtained 
using `PalusLimit`, plus two additional finer partitions. 

```jldoctest
using CausalityTools
x = rand(1000) # our time series
D = 3 # dimension of the desired state space reconstruction
heuristic = ExtendedPalusLimit(ext_finer = 2)
estimate_partition(x, D, heuristic)

# output

3-element Array{RectangularBinning,1}:
 RectangularBinning(5)
 RectangularBinning(6)
 RectangularBinning(7)
```
"""
function estimate_partition(x::AbstractVector{T}, dim::Int, heuristic::BinningHeuristic) where T end

function estimate_partition(x::AbstractVector{T}, dim::Int, heuristic::PalusLimit) where T
    N = length(x)
    n_subdivs = min(floor(Int, N^(1/(dim+1))), heuristic.max_subdivs)
    return RectangularBinning(n_subdivs)
end

function estimate_partition(x::AbstractVector{T}, dim::Int, heuristic::ExtendedPalusLimit) where T
    N = length(x)
    n_subdivs = min(floor(Int, N^(1/(dim+1))), heuristic.max_subdivs)
    
    min_subdivs = max(heuristic.min_subdivs, n_subdivs - heuristic.ext_coarser)
    max_subdivs = min(heuristic.max_subdivs, n_subdivs + heuristic.ext_finer)
    
    if min_subdivs == max_subdivs
        return RectangularBinning(min_subdivs)
    else
        return [RectangularBinning(bs) for bs in min_subdivs:max_subdivs]
    end
end
