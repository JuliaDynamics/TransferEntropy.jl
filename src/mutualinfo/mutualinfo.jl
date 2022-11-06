using Entropies: ProbabilitiesEstimator, IndirectEntropy, Entropy, Shannon

export MutualInformation
export mutualinfo

""" The supertype of all dedicated mutual information estimators """
abstract type MutualInformation end

"""
    mutualinfo([e::Entropy,] est::ProbabilitiesEstimator, x, y)

Estimate ``I(x; y)``, the mutual information between `x` and `y`, by a sum of marginal
entropies of type `e`, using the provided [`ProbabilitiesEstimator`](@ref).

If the entropy type is not specified, then `Shannon(; base = 2)` is used.

## Description

Mutual information ``I`` between ``X`` and ``Y``
is defined as

```math
I(X; Y) =
\\sum_{y \\in Y} \\sum_{x \\in X} p(x, y) \\log \\left( \\dfrac{p(x, y)}{p(x)p(y)} \\right)
```

Here, we rewrite this expression as the sum of the marginal entropies, and extend the
definition of ``I`` to use generalized entropies

```math
I^{q}(X; Y) = H^{q}(X) + H^{q}(Y) - H^{q}(X, Y),
```

where ``H^{q}(\\cdot)`` is the generalized entropy with parameter ``q``. The meaning of `q`
depends on `e`, and `q = 1` for Shannon entropy.
"""
function mutualinfo(e::Entropy, est::ProbabilitiesEstimator,
        x::Vector_or_Dataset,
        y::Vector_or_Dataset)
    X = entropy(e, Dataset(x), est)
    Y = entropy(e, Dataset(y), est)
    XY = entropy(e, Dataset(x, y), est)
    MI = X + Y - XY
end
mutualinfo(est::ProbabilitiesEstimator, x::Vector_or_Dataset, y::Vector_or_Dataset) =
    mutualinfo(Shannon(; base = 2), est, x, y)
mutualinfo(e::Entropy, x::Vector_or_Dataset, y::Vector_or_Dataset) =
    error("Estimator missing. Please provide a valid estimator as the second argument.")

"""
    mutualinfo(e::IndirectEntropy, x, y)

Estimate ``I(x; y)``, the  mutual information between `x` and `y`, by a sum of marginal
entropies (whose type is dictated by `e`), using the provided [`IndirectEntropy`](@ref)
estimator.

Both `x` and `y` can be vectors or (potentially multivariate) [`Dataset`](@ref)s.
"""
function mutualinfo(e::IndirectEntropy, x::Vector_or_Dataset, y::Vector_or_Dataset)
    X = entropy(e, Dataset(x))
    Y = entropy(e, Dataset(y))
    XY = entropy(e, Dataset(x, y))
    MI = X + Y - XY
end

include("estimators/estimators.jl")
