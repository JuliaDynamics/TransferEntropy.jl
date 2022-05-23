export mutualinfo, cmi, Kraskov1, Kraskov2

abstract type MutualInformationEstimator <: EntropyEstimator end

"""
    mutualinfo(x, y, est; base = 2, q = 1)

Estimate mutual information between `x` and `y`, ``I^{q}(x; y)``, using the provided 
entropy/probability estimator `est` from Entropies.jl, and Rényi entropy of order `q`
(defaults to `q = 1`, which is the Shannon entropy), with logarithms to the given `base`.

Both `x` and `y` can be vectors or (potentially multivariate) [`Dataset`](@ref)s.

Worth highlighting here are the estimators that compute entropies _directly_, e.g.
nearest-neighbor based methhods. The choice is between naive 
estimation using the [`KozachenkoLeonenko`](@ref) or [`Kraskov`](@ref) entropy estimators, 
or the improved [`Kraskov1`](@ref) and [`Kraskov2`](@ref) dedicated ``I`` estimators. The 
latter estimators reduce bias compared to the naive estimators.

**Note**: only Shannon entropy is possible to use for nearest neighbor estimators, so the 
keyword `q` cannot be provided; it is hardcoded as `q = 1`. 

## Description

Mutual information ``I`` between ``X`` and ``Y`` 
is defined as 

```math
I(X; Y) = \\sum_{y \\in Y} \\sum_{x \\in X} p(x, y) \\log \\left( \\dfrac{p(x, y)}{p(x)p(y)} \\right)
```

Here, we rewrite this expression as the sum of the marginal entropies, and extend the 
definition of ``I`` to use generalized Rényi entropies

```math
I^{q}(X; Y) = H^{q}(X) + H^{q}(Y) - H^{q}(X, Y),
```

where ``H^{q}(\\cdot)`` is the generalized Renyi entropy of order ``q``, i.e., the
`genentropy` function from Entropies.jl.
"""
function mutualinfo end 

mutualinfo(x::Vector_or_Dataset, y::Vector_or_Dataset) = 
    error("Estimator missing. Please provide a valid estimator as the third argument.")

function mutualinfo(x::Vector_or_Dataset, y::Vector_or_Dataset, est; base = 2, q = 1)
    X = genentropy(Dataset(x), est; base = base, q = q)
    Y = genentropy(Dataset(y), est; base = base, q = q)
    XY = genentropy(Dataset(x, y), est; base = base, q = q)
    MI = X + Y - XY 
end

"""
    cmi(x::Vector_or_Dataset, y::Vector_or_Dataset, z::Vector_or_Dataset, est; 
        base = 2, q = 1)
    
Compute the conditional mutual information (CMI), ``I(X; Y | Z)``, between variables
`x`, `y`, and `z` (each either a univariate `Vector` or multivariate `Dataset`).
"""
function cmi(x::Vector_or_Dataset, y::Vector_or_Dataset, z::Vector_or_Dataset, est; 
        base = 2, q = 1)
    mutualinfo(x, Dataset(y, z), est; base = base, q = q) -
        mutualinfo(x, z, est; base = base, q = q)
end

include("nearestneighbor.jl")