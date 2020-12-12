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


""" 
Abstract types for transfer entropy estimators that are not already implemented as basic 
entropy estimators in Entropies.jl, but needs some kind of special treatment to  work 
for transfer entropy. """
abstract type TransferEntropyEstimator <: EntropyEstimator end

"""
## Transfer entropy

Transfer entropy between two simultaneously measured scalar time series ``s(n)`` and ``t(n)``,  
``s(n) = \\{ s_1, s_2, \\ldots, s_N \\} `` and ``t(n) = \\{ t_1, t_2, \\ldots, t_N \\} ``, is
is defined as 

```math 
TE(s \\to t) = \\sum_i p(s_i, t_i, t_{i+\\eta}) \\log \\left( \\dfrac{p(t_{i+\\eta} | t_i, s_i)}{p(t_{i+\\eta} | t_i)} \\right)
```

Including more than one historical/future value can be done by defining the vector-valued
time series
``\\mathcal{T}^{(d_{\\mathcal T}, \\eta_{\\mathcal T})} = \\{t_i^{(d_{\\mathcal T}, \\eta_{\\mathcal T})} \\}_{i=1}^{N}``, 
``T^{(d_T, \\tau_T)} = \\{t_i^{(d_T, \\tau_T)} \\}_{i=1}^{N}``, 
``S^{(d_S, \\tau_S)} = \\{s_i^{(d_T, \\tau_T)} \\}_{i=1}^{N}``,  and 
``C^{(d_C, \\tau_C)} = \\{s_i^{(d_C, \\tau_C)} \\}_{i=1}^{N}``.

The `N` state vectors for each marginal are either 
- uniform, of the form ``x_{i}^{(d_X, \\tau_X)} = (x_i, x_{i-\\tau}, x_{i-2\\tau}, \\ldots x_{i-(dX - 1)\\tau_X})``, 
    with equally spaced state vector entries.
- non-uniform, of the form ``x_{i}^{(d_X, \\tau_X)} = (x_i, x_{i-\\tau_1}, x_{i-\\tau_2}, \\ldots x_{i-\\tau_{dX}})``,
    with non-equally spaced state vector entries ``\\tau_1, \\tau_2, \\ldots, \\tau_{dX}``,
    which can freely chosen.

The 
``d_T``-dimensional, ``d_S``-dimensional and ``d_C``-dimensional state vectors 
comprising ``T``, ``S`` and ``C`` are constructed with embedding lags 
``\\tau_T``, ``\\tau_S``, and ``\\tau_C``, respectively. 
The ``d_{\\mathcal T}``-dimensional 
future states ``\\mathcal{T}^{(d_{\\mathcal T}, \\eta_{\\mathcal T})}``
are constructed with prediction lag ``\\eta_{\\mathcal T}`` (i.e. predictions go from 
present/past states to future states spanning a maximum of 
``d_{\\mathcal T} \\eta_{\\mathcal T}`` time steps ).
*Note: in the original transfer entropy paper, only the historical states are defined as 
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
`α` as

```math
TE^{\\alpha}(s \\to t) = H^{\\alpha}(\\mathcal T, T) + H^{\\alpha}(T, S) - H^{\\alpha}(T) - H^{\\alpha}(\\mathcal T, T, S),
```

```math
TE^{\\alpha}(s \\to t | c) = H^{\\alpha}(\\mathcal T, T, C) + H^{\\alpha}(T, S, C) - H^{\\alpha}(T, C) - H^{\\alpha}(\\mathcal T, T, S, C),
```

where ``H^{\\alpha}(\\cdot)`` is the generalized Renyi entropy of order ``\\alpha``. 

## General interface 

    transferentropy(s, t, [c], est; base = 2, α = 1, 
        τT = -1, τS = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1, [τC = -1, dC = 1])

Estimate transfer entropy from source `s` to target `t`, ``TE^{\\alpha}(s \\to t)``, using the 
provided entropy/probability estimator `est` and Rényi entropy of order-`α` (defaults to `α = 1`, 
which is the Shannon entropy), with logarithms to the given `base`. Optionally, condition 
on `c` and estimate the conditional transfer entropy ``TE^{\\alpha}(s \\to t | c)``. 

Parameters for embedding lags `τT`, `τS`, `τC`, the `η𝒯` (prediction lag), and 
the embedding dimensions `dT`, `dS`, `dC`, `d𝒯` have meanings as explained above. 
Here, the convention is to use negative lags to indicate embedding delays for past state 
vectors (for the ``T``, ``S`` and ``C`` marginals), and positive lags to indicate embedding 
delays for future state vectors (for the ``\\mathcal T`` marginal). 

Default embedding values use scalar time series for each marginal. Hence, the value(s) of 
`τT`, `τS` or `τC` affect the estimated ``TE`` only if the corresponding dimension(s) 
`dT`, `dS` or `dC` are larger than `1`.

The input series `s`, `t`, and `c` must be equal-length real-valued vectors of length `N`.

## Binning based

    transferentropy(s, t, [c], est::VisitationFrequency{RectangularBinning}; base = 2, α = 1, ...)

Estimate transfer entropy using visitation frequencies over a rectangular binning.

See also: [`VisitationFrequency`](@ref), [`RectangularBinning`](@ref).

## Nearest neighbor based 

    transferentropy(s, t, [c], est::Kraskov; base = 2, ...)
    transferentropy(s, t, [c], est::KozachenkoLeonenko; base = 2, ...)

Estimate ``TE^{1}(s \\to t)``/``TE^{1}(s \\to t | c)`` using naive nearest neighbor estimators.

*Note: only Shannon entropy is possible to use for nearest neighbor estimators, so the 
keyword `α` cannot be provided; it is hardcoded as `α = 1`*. 

See also [`Kraskov`](@ref), [`KozachenckoLeonenko`](@ref).

## Kernel density based 

    transferentropy(s, t, [c], est::NaiveKernel{Union{TreeDistance, DirectDistance}}; 
        base = 2, α = 1,  ...)

Estimate ``TE^{\\alpha}(s \\to t)``/``TE^{\\alpha}(s \\to t | c)`` using naive kernel density estimation of 
probabilities.

See also [`NaiveKernel`](@ref), [`TreeDistance`](@ref), [`DirectDistance`](@ref).

## Instantenous Hilbert amplitudes/phases 

    transferentropy(s, t, [c], est::Hilbert; base = 2, α = 1,  ...)

Estimate ``TE^{\\alpha}(s \\to t)``/``TE^{\\alpha}(s \\to t | c)`` by first applying the Hilbert transform 
to `s`, `t` (`c`) and then estimating transfer entropy.

See also [`Hilbert`](@ref), [`Amplitude`](@ref), [`Phase`](@ref).

## Symbolic/permutation

    transferentropy(s, t, [c], est::SymbolicPermutation; 
        base = 2, α = 1, m::Int = 3, τ::Int = 1, ...)
    transferentropy!(symb_s, symb_t, s, t, [c], est::SymbolicPermutation; 
        base = 2, α = 1, m::Int = 3, τ::Int = 1, ...)

Estimate ``TE^{\\alpha}(s \\to t)``/``TE^{\\alpha}(s \\to t | c)`` using permutation entropies. This is done 
by first symbolizing the input series `s` and `t` (and `c`; all of length `N`) using motifs of 
size `m` and a time delay of `τ`. The series of motifs are encoded as integer symbol time 
series preserving the permutation information. These symbol time series are embedded as 
usual, and transfer entropy is computed from marginal entropies of that generalized embedding.

Optionally, provide pre-allocated (integer) symbol vectors `symb_s` and `symb_t` (and `symb_c`),
where `length(symb_s) == length(symb_t) == length(symb_c) == N - (est.m-1)*est.τ`. This is useful for saving 
memory allocations for repeated computations.

See also [`SymbolicPermutation`](@ref).

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
# = (t_i, t_{i-2}, t_{i-4}, \\ldots t_{i-6τ})
transferentropy(x, y, est, dT = 4, τT = -2)

# Non-uniform delay vectors when `τT` is a vector of integers
# Here t_{i}^{(dT, τT)} = (t_i, t_{i+τ_{1}}, t_{i+τ_{2}}, \\ldots t_{i+τ_{dT}})
# = (t_i, t_{i-7}, t_{i-25})
transferentropy(x, y, est, dT = 3, τT = [0, -7, -25])
```

Logarithm bases and the order of the Rényi entropy can also be tuned:

```julia
x, y = rand(100), rand(100)
est = NaiveKernel(0.3)
transferentropy(x, y, est, base = MathConstants.e, α = 2) # TE in nats, order-2 Rényi entropy
```
"""
function transferentropy end 
function transferentropy! end

# estimate transfer entropy from marginal entropies, as described in docstring
function transferentropy(joint, ST, T𝒯, T, est; base = 2, α = 1)
    te = genentropy(T𝒯, est, base = base, α = α) +
        genentropy(ST, est, base = base, α = α) -
        genentropy(T, est, base = base, α = α) -
        genentropy(joint, est, base = base, α = α)
end

# TODO: estimate using mutual information decomposition, 
# function transferentropy(marginal1, marginal2, est; base = 2, α = 1)


# Estimate transfer entropy from time series by first embedding them and getting required 
# marginals.
function transferentropy(s, t, est; base = 2, α = 1, 
        τT = -1, τS = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1)
    
    emb = EmbeddingTE(τT = τT, τS = τS, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯)
    joint, ST, T𝒯, T = get_marginals(s, t, emb)

    transferentropy(joint, ST, T𝒯, T, est; base = base, α = α)

end

function transferentropy(s, t, c, est; base = 2, α = 1, 
        τT = -1, τS = -1, τC = -1, η𝒯 = 1, dT = 1, dS = 1, dC = 1, d𝒯 = 1)
    
    emb = EmbeddingTE(τT = τT, τS = τS, τC = τC, η𝒯 = η𝒯, dT = dT, dS = dS, dC = dC, d𝒯 = d𝒯)
    joint, ST, T𝒯, T = get_marginals(s, t, c, emb)

    transferentropy(joint, ST, T𝒯, T, est; base = base, α = α)
end

transferentropy(s::Vector{<:Real}, t::Vector{<:Real}) = 
    error("Estimator missing. Please provide a valid estimator as the third argument.")

transferentropy(s::Vector{<:Real}, t::Vector{<:Real}, c::Vector{<:Real}) = 
    error("Estimator missing. Please provide a valid estimator as the fourth argument.")


include("symbolic.jl")
include("binning_based.jl")
include("hilbert.jl")
include("nearestneighbor.jl")