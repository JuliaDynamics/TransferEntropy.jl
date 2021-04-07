include("utils.jl")

export transferentropy, transferentropy!

function get_marginals(s, t, emb::EmbeddingTE)
    pts, vars, τs, js = te_embed(s, t, emb)
    
    # Get marginals
    ST = pts[:, [vars.S; vars.T]]
    T𝒯 = pts[:, [vars.𝒯; vars.T]]
    T = pts[:, vars.T]
    joint = pts

    return joint, ST, T𝒯, T
end

function get_marginals(s, t, c, emb::EmbeddingTE)
    pts, vars, τs, js = te_embed(s, t, c, emb)
    
    # Get marginals
    ST = pts[:, [vars.S; vars.T; vars.C]]
    T𝒯 = pts[:, [vars.𝒯; vars.T; vars.C]]
    T = pts[:, [vars.T; vars.C]]
    joint = pts

    return joint, ST, T𝒯, T
end

# map a set of pre-embedded points to the correct marginals for transfer entropy computation
function get_marginals(pts, emb::TEVars)    
    # Get marginals
    ST = pts[:, [vars.S; vars.T]]
    T𝒯 = pts[:, [vars.𝒯; vars.T]]
    T = pts[:, vars.T]
    joint = pts

    return joint, ST, T𝒯, T
end


""" 
Abstract types for transfer entropy estimators that are not already implemented as basic 
entropy estimators in Entropies.jl, but needs some kind of special treatment to  work 
for transfer entropy. """
abstract type TransferEntropyEstimator <: EntropyEstimator end

"""
## Transfer entropy (general interface)

    transferentropy(s, t, [c], est; base = 2, q = 1, 
        τT = -1, τS = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1, [τC = -1, dC = 1]) → Float64

Estimate transfer entropy from source `s` to target `t`, ``TE^{q}(s \\to t)``, using the 
provided entropy/probability estimator `est` and Rényi entropy of order-`q` (defaults to `q = 1`, 
which is the Shannon entropy), with logarithms to the given `base`. Optionally, condition 
on `c` and estimate the conditional transfer entropy ``TE^{q}(s \\to t | c)``. 

Parameters for embedding lags `τT`, `τS`, `τC` (must be negative), the prediction lag `η𝒯` 
(must be positive), and the embedding dimensions `dT`, `dS`, `dC`, `d𝒯` have meanings as 
explained above. Here, the convention is to use negative lags to indicate embedding delays 
for past state vectors (for the ``T``, ``S`` and ``C`` marginals), and positive lags to 
indicate embedding delays for future state vectors (for the ``\\mathcal T`` marginal). 

Default embedding values use scalar time series for each marginal. Hence, the value(s) of 
`τT`, `τS` or `τC` affect the estimated ``TE`` only if the corresponding dimension(s) 
`dT`, `dS` or `dC` are larger than `1`.

The input series `s`, `t`, and `c` must be equal-length real-valued vectors of length `N`.

## Binning based

    transferentropy(s, t, [c], est::VisitationFrequency{RectangularBinning}; base = 2, q = 1, ...) → Float64

Estimate ``TE^{q}(s \\to t)`` or ``TE^{q}(s \\to t | c)`` using visitation frequencies over a rectangular binning.

    transferentropy(s, t, [c], est::TransferOperator{RectangularBinning}; base = 2, q = 1, ...) → Float64

Estimate ``TE^{q}(s \\to t)`` or ``TE^{q}(s \\to t | c)`` using an approximation to the transfer operator over rectangular 
binning.

See also: [`VisitationFrequency`](@ref), [`RectangularBinning`](@ref).

## Nearest neighbor based 

    transferentropy(s, t, [c], est::Kraskov; base = 2, ...) → Float64
    transferentropy(s, t, [c], est::KozachenkoLeonenko; base = 2, ...) → Float64

Estimate ``TE^{1}(s \\to t)`` or ``TE^{1}(s \\to t | c)`` using naive nearest neighbor estimators.

*Note: only Shannon entropy is possible to use for nearest neighbor estimators, so the 
keyword `q` cannot be provided; it is hardcoded as `q = 1`*. 

See also [`Kraskov`](@ref), [`KozachenckoLeonenko`](@ref).

## Kernel density based 

    transferentropy(s, t, [c], est::NaiveKernel{Union{TreeDistance, DirectDistance}}; 
        base = 2, q = 1,  ...) → Float64

Estimate ``TE^{q}(s \\to t)`` or ``TE^{q}(s \\to t | c)`` using naive kernel density estimation of 
probabilities.

See also [`NaiveKernel`](@ref), [`TreeDistance`](@ref), [`DirectDistance`](@ref).

## Instantenous Hilbert amplitudes/phases 

    transferentropy(s, t, [c], est::Hilbert; base = 2, q = 1,  ...) → Float64

Estimate ``TE^{q}(s \\to t)`` or ``TE^{q}(s \\to t | c)`` by first applying the Hilbert transform 
to `s`, `t` (`c`) and then estimating transfer entropy.

See also [`Hilbert`](@ref), [`Amplitude`](@ref), [`Phase`](@ref).

## Symbolic/permutation

    transferentropy(s, t, [c], est::SymbolicPermutation; 
        base = 2, q = 1, m::Int = 3, τ::Int = 1, ...) → Float64
    transferentropy!(symb_s, symb_t, s, t, [c], est::SymbolicPermutation; 
        base = 2, q = 1, m::Int = 3, τ::Int = 1, ...) → Float64

Estimate ``TE^{q}(s \\to t)`` or ``TE^{q}(s \\to t | c)`` using permutation entropy. This is done 
by first symbolizing the input series `s` and `t` (and `c`; all of length `N`) using motifs of 
size `m` and a time delay of `τ`. The series of motifs are encoded as integer symbol time 
series preserving the permutation information. These symbol time series are embedded as 
usual, and transfer entropy is computed from marginal entropies of that generalized embedding.

Optionally, provide pre-allocated (integer) symbol vectors `symb_s` and `symb_t` (and `symb_c`),
where `length(symb_s) == length(symb_t) == length(symb_c) == N - (est.m-1)*est.τ`. This is useful for saving 
memory allocations for repeated computations.

See also [`SymbolicPermutation`](@ref).
## Description

Transfer entropy between two simultaneously measured scalar time series ``s(n)`` and ``t(n)``,  
``s(n) = \\{ s_1, s_2, \\ldots, s_N \\} `` and ``t(n) = \\{ t_1, t_2, \\ldots, t_N \\} ``, is
is defined as 

```math 
TE(s \\to t) = \\sum_i p(s_i, t_i, t_{i+\\eta}) \\log \\left( \\dfrac{p(t_{i+\\eta} | t_i, s_i)}{p(t_{i+\\eta} | t_i)} \\right)
```

### Generalized embeddings

By defining the vector-valued time series

- ``\\mathcal{T}^{(d_{\\mathcal T}, \\eta_{\\mathcal T})} = \\{t_i^{(d_{\\mathcal T}, \\eta_{\\mathcal T})} \\}_{i=1}^{N}``, 
-``T^{(d_T, \\tau_T)} = \\{t_i^{(d_T, \\tau_T)} \\}_{i=1}^{N}``, 
- ``S^{(d_S, \\tau_S)} = \\{s_i^{(d_T, \\tau_T)} \\}_{i=1}^{N}``,  and 
- ``C^{(d_C, \\tau_C)} = \\{s_i^{(d_C, \\tau_C)} \\}_{i=1}^{N}``,

it is possible to include more than one historical/future value for each marginal.

The ``d_T``-dimensional, ``d_S``-dimensional and ``d_C``-dimensional state vectors 
comprising ``T``, ``S`` and ``C`` are constructed with embedding lags 
``\\tau_T``, ``\\tau_S``, and ``\\tau_C``, respectively. 
The ``d_{\\mathcal T}``-dimensional 
future states ``\\mathcal{T}^{(d_{\\mathcal T}, \\eta_{\\mathcal T})}``
are constructed with prediction lag ``\\eta_{\\mathcal T}`` (i.e. predictions go from 
present/past states to future states spanning a maximum of 
``d_{\\mathcal T} \\eta_{\\mathcal T}`` time steps).
*Note: in Schreiber's paper, only the historical states are defined as 
potentially higher-dimensional, while the future states are always scalar.*

The non-conditioned and conditioned generalized forms of the transfer entropy are then

```math 
TE(s \\to t) = \\sum_i p(S,T, \\mathcal{T}) \\log \\left( \\dfrac{p(\\mathcal{T} | T, S)}{p(\\mathcal{T} | T)} \\right)
```

```math 
TE(s \\to t | c) = \\sum_i p(S,T, \\mathcal{T}, C) \\log \\left( \\dfrac{p(\\mathcal{T} | T, S, C)}{p(\\mathcal{T} | T, C)} \\right)
```

## Estimation 

Transfer entropy is here estimated by rewriting the above expressions as a sum of marginal 
entropies, and extending the definitions above to use Rényi generalized entropies of order 
`q` as

```math
TE^{q}(s \\to t) = H^{q}(\\mathcal T, T) + H^{q}(T, S) - H^{q}(T) - H^{q}(\\mathcal T, T, S),
```

```math
TE^{q}(s \\to t | c) = H^{q}(\\mathcal T, T, C) + H^{q}(T, S, C) - H^{q}(T, C) - H^{q}(\\mathcal T, T, S, C),
```

where ``H^{q}(\\cdot)`` is the generalized Renyi entropy of order ``q``. 

### Uniform vs. non-uniform embeddings

The `N` state vectors for each marginal are either 

- uniform, of the form ``x_{i}^{(d_X, \\omega_X)} = (x_i, x_{i+\\omega}, x_{i+2\\omega}, \\ldots x_{i+(dX - 1)\\omega_X})``, 
    with equally spaced state vector entries. *Remember: When constructing marginals for ``T``, ``S`` and ``C``, 
    we need ``\\omega \\leq 0`` to get present/past values, while ``\\omega > 0`` is necessary to get future states 
    when constructing ``\\mathcal{T}``.
- non-uniform, of the form ``x_{i}^{(d_X, \\omega_X)} = (x_i, x_{i+\\omega_1}, x_{i+\\omega_2}, \\ldots x_{i+\\omega_{dX}})``,
    with non-equally spaced state vector entries ``\\omega_1, \\omega_2, \\ldots, \\omega_{dX}``,
    which can be freely chosen. *Remember: When constructing marginals for ``T``, ``S`` and ``C``, 
    we need ``\\omega_i \\leq 0`` for all ``\\omega_i`` to get present/past values, while ``\\omega_i > 0`` for all ``\\omega_i`` 
    is necessary to get future states when constructing ``\\mathcal{T}``.*

## Examples

Default estimation (scalar marginals): 

```julia
# Symbolic estimator, motifs of length 4, uniform delay vectors with lag 1
est = SymbolicPermutation(m = 4, τ = 1) 

x, y = rand(100), rand(100)
transferentropy(x, y, est)
```

Increasing the dimensionality of the ``T`` marginal (present/past states of the target 
variable):

```julia
# Binning-based estimator
est = VisitationFrequency(RectangularBinning(4)) 
x, y = rand(100), rand(100)

# Uniform delay vectors when `τT` is an integer (see explanation above)
# Here t_{i}^{(dT, τT)} = (t_i, t_{i+τ}, t_{i+2τ}, \\ldots t_{i+(dT-1)τ})
# = (t_i, t_{i-2}, t_{i-4}, \\ldots t_{i-6τ}), so we need zero/negative values for `τT`.
transferentropy(x, y, est, dT = 4, τT = -2)

# Non-uniform delay vectors when `τT` is a vector of integers
# Here t_{i}^{(dT, τT)} = (t_i, t_{i+τ_{1}}, t_{i+τ_{2}}, \\ldots t_{i+τ_{dT}})
# = (t_i, t_{i-7}, t_{i-25}), so we need zero/negative values for `τT`.
transferentropy(x, y, est, dT = 3, τT = [0, -7, -25])
```

Logarithm bases and the order of the Rényi entropy can also be tuned:

```julia
x, y = rand(100), rand(100)
est = NaiveKernel(0.3)
transferentropy(x, y, est, base = MathConstants.e, q = 2) # TE in nats, order-2 Rényi entropy
```
"""
function transferentropy end 
function transferentropy! end

# estimate transfer entropy from marginal entropies, as described in docstring
function _transferentropy(joint, ST, T𝒯, T, est::Est; base = 2, q = 1)
    te = genentropy(T𝒯, est, base = base, q = q) +
        genentropy(ST, est, base = base, q = q) -
        genentropy(T, est, base = base, q = q) -
        genentropy(joint, est, base = base, q = q)
end

# TODO: estimate using mutual information decomposition, 
# function transferentropy(marginal1, marginal2, est; base = 2, q = 1)


# Estimate transfer entropy from time series by first embedding them and getting required 
# marginals.
function transferentropy(s, t, est::Est; base = 2, q = 1, 
        τT = -1, τS = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1)
    
    emb = EmbeddingTE(τT = τT, τS = τS, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯)
    joint, ST, T𝒯, T = get_marginals(s, t, emb)

    _transferentropy(joint, ST, T𝒯, T, est; base = base, q = q)
end

function transferentropy(s, t, c, est::Est; base = 2, q = 1, 
        τT = -1, τS = -1, τC = -1, η𝒯 = 1, dT = 1, dS = 1, dC = 1, d𝒯 = 1)
    
    emb = EmbeddingTE(τT = τT, τS = τS, τC = τC, η𝒯 = η𝒯, dT = dT, dS = dS, dC = dC, d𝒯 = d𝒯)
    joint, ST, T𝒯, T = get_marginals(s, t, c, emb)

    _transferentropy(joint, ST, T𝒯, T, est; base = base, q = q)
end

transferentropy(s::AbstractVector{<:Real}, t::AbstractVector{<:Real}) = 
    error("Estimator missing. Please provide a valid estimator as the third argument.")

transferentropy(s::AbstractVector{<:Real}, t::AbstractVector{<:Real}, c::AbstractVector{<:Real}) = 
    error("Estimator missing. Please provide a valid estimator as the fourth argument.")


include("symbolic.jl")
include("binning_based.jl")
include("hilbert.jl")
include("nearestneighbor.jl")