function rc(x::Union{AbstractDataset, Vector{T}, Vector{Vector{T}}},
        dim::Union{Int, AbstractVector{Int}, OptimiseDim}, 
        τ::Union{Int, AbstractVector{Int}, OptimiseDelay}, forward = false) where T <: Number
    
    if dim isa OptimiseDim
        if x isa Vector{T} where T <: Number
            dim = optimal_dimension(x, dim)
        elseif x isa Vector{Vector{T}} where T <: Number
            dim = [optimal_dimension(ts, dim) for ts in x]
        elseif x isa AbstractDataset
            dim = [optimal_dimension(x[:, i], dim) for i = 1:size(x, 2)]
        end
    end

    if τ isa OptimiseDelay
        # only optimise if marginal has dimension greater than 1 (the first component 
        # is always constructed with the instantaneous lag, so no need to search for lags for that)
        if dim > 1 
            if x isa Vector{T} where T <: Number
                forward ? τ = optimal_delay(x, τ) : τ = -optimal_delay(x, τ)
            elseif x isa Vector{Vector{T}} where T <: Number
                forward ? τ = [optimal_delay(ts, τ) for ts in x] : τ = -[optimal_delay(ts, τ) for ts in x]
            elseif x isa AbstractDataset
                forward ? τ = [optimal_delay(x[:, i], τ) for i = 1:size(x, 2)] : -[optimal_delay(x[:, i], τ) for i = 1:size(x, 2)]
            end
        else 
            τ = -1
        end
    end
    
    if x isa Vector{T} where T <: Number
        # Fixed dimension, fixed lag
        if dim isa Int && τ isa Int            
            pos = [1 for x in dim:-1:1]
            if τ > 0
                lags = [d*τ for d in dim:-1:1]
            elseif τ <= 0 && forward
                lags = [d*τ for d in dim:-1:1]
            elseif τ <= 0 && !forward
                lags = [d*τ for d in 0:(dim-1)]
            end
        end
        
        # Fixed dimension, multiple lags (number of lags must match dimension)
        if dim isa Int && length(τ) > 1
            length(τ) == dim || throw(ArgumentError("length(τ) must equal dim if multiple lags are specified manually (got length(τ) = $(length(τ)), dim=$(dim))"))
            length(unique(τ)) == length(τ) || throw(ArgumentError("Tried to specify reconstruction lags manually, but there are repeated lags. Provide unique lags."))
            pos = pos = [1 for x in dim:-1:1]
            return pos, τ
        end
    end

    # Multiple time series input
    if (x isa Vector{Vector{T}} where T <: Number) || (x isa AbstractDataset)
        if x isa AbstractDataset 
            N = length(x)
        elseif x isa Vector{Vector{T}} where T <: Number
            N = size(x, 1)
        end


        if dim isa Int
            dim % N == 0 || throw(ArgumentError("If using multiple (`N` different) time series in a marginal, each time series is lagged `dim/N` times. Hence, `dim` must be a multiple of `N`."))

            lD = Int(dim / N) # "local" reconstruction dimension for this time series
                
            pos = Vector{Int}(undef, 0)

            for i = 1:N
                append!(pos, repeat([i], lD))
            end
            
            if τ isa Int
                if τ > 0
                    llags = [d*τ for d in lD:-1:1]
                elseif τ <= 0 && forward
                    llags = [d*τ for d in lD:-1:1]
                elseif τ <= 0 && !forward
                    llags = [d*τ for d in 0:(lD - 1)]
                end
                lags = repeat(llags, Int(dim/lD))
            elseif τ isa AbstractVector{Int}
                length(τ) == N || throw(ArgumentError("Tried using `N = $N` different time series in a marginal, but $(length(τ)) reconstruction lags were provided. The number of lags must equal `N`"))
                lags = Vector{Int}(undef, 0)
                for i = 1:length(x)
                    if τ[i] > 0
                        llags = [d*τ[i] for d in lD:-1:1]
                    elseif τ[i] <= 0 && forward
                        llags = [d*τ[i] for d in 0:(lD - 1)]
                    elseif τ[i] <= 0 && !forward
                        llags = [d*τ[i] for d in 0:(lD - 1)]
                    end
                    append!(lags, llags)
                end
            end
        end
        
        if dim isa AbstractVector{Int}
            length(dim) == N || throw(ArgumentError("There must be precisely one dimension specification per time series. Got $(length(dim)) specifications for $N time series."))
            pos = Vector{Int}(undef, 0)
            lags = Vector{Int}(undef, 0)

            for (i, lD) in enumerate(dim)
                append!(pos, repeat([i], lD))
                
                if τ isa Int
                    if τ > 0
                        llags = [d*τ for d in lD:-1:1]
                    elseif τ <= 0 && !forward
                        llags = [d*τ for d in lD:-1:1]
                    elseif τ <= 0 && forward
                        llags = [d*τ for d in 0:(lD - 1)]
                    end
                    append!(lags, llags)
                elseif τ isa AbstractVector{Int}
                    length(τ) == N || throw(ArgumentError("Tried using `N = $N` different time series in a marginal, but $(length(τ)) reconstruction lags were provided. The number of lags must equal `N`"))
                    if τ[i] > 0
                        llags = [d*τ[i] for d in lD:-1:1]
                    elseif τ[i] <= 0 && forward
                        llags = [d*τ[i] for d in lD:-1:1]
                    elseif τ[i] <= 0 && !forward
                        llags = [d*τ[i] for d in 0:(lD - 1)]
                    end
                    append!(lags, llags)
                end
            end
        end
    end
    return pos, lags
end


"""
    EmbeddingTE(dS = 1, dT = 1, d𝒯 = 1, dC = 1, 
        τS = -1, τT = -1, η = 1, τC = -1)

Embedding parameters for transfer entropy analysis. 

## Convention for generalised delay reconstruction

This struct contains instructions for transfer entropy computations using the following convention.
Let ``S`` be time series for the source variable, ``T`` be the time series for the target variable and 
``C`` the time series for any conditional variable. To compute transfer entropy, we need the 
following marginals:

```math
\\begin{align}
\\mathcal{T}^{(d_{\\mathcal{T}})} &= \\{(T(t+\\eta^{d_{\\mathcal{T}}}), \\ldots, T(t+\\eta^2), T(t+\\eta^1) \\} \\\\
T^{(d_{T})} &= \\{ (T(t), T(t-\\tau^1_{T}), T(t-\\tau^2_{T}), \\ldots, T(t - \\tau^{d_{T} - 1}_{T})) \\} \\\\
S^{(d_{S})} &= \\{ (S(t), S(t-\\tau^1_{S}), S(t-\\tau^2_{S}), \\ldots, S(t-\\tau^{d_{S} - 1}_{S})) \\} \\\\
C^{(d_{C})} &= \\{ (C(t), C(t-\\tau^1_{C}), C(t-\\tau^2_{C}), \\ldots, C(t-\\tau^{d_{C} - 1}_{C})) \\}
\\end{align}
```

and combined, we get the generalised delay reconstruction ``\\mathbb{E} = (\\mathcal{T}^{(d_{\\mathcal{T}})}, T^{(d_{T})}, S^{(d_{S})}, C^{(d_{C})})``. Transfer entropy is then computed as 

```math
\\begin{align}
TE_{S \\rightarrow T | C} = \\int_{\\mathbb{E}} P(\\mathcal{T}, T, S, C) \\log_{b}{\\left(\\frac{P(\\mathcal{T} | T, S, C)}{P(\\mathcal{T} | T, C)}\\right)},
\\end{align}
```

or, if conditionals are not relevant,

```math
\\begin{align}
TE_{S \\rightarrow T} = \\int_{\\mathbb{E}} P(\\mathcal{T}, T, S) \\log_{b}{\\left(\\frac{P(\\mathcal{T} | T, S)}{P(\\mathcal{T} | T)}\\right)},
\\end{align}
```

Here, 

- ``\\mathcal{T}`` denotes the ``d_{\\mathcal{T}}``-dimensional set of vectors furnishing the future states of ``T``,
- ``T`` denotes the ``d_{T}``-dimensional set of vectors furnishing the past and present states of ``T``, 
- ``S`` denotes the ``d_{S}``-dimensional set of vectors furnishing the past and present of ``S``, and 
- ``C`` denotes the ``d_{C}``-dimensional set of vectors furnishing the past and present of ``C``.

## Keyword arguments 

### Specifying dimensions for generalised delay reconstructions of marginals

`dS`, `dT`, `d𝒯`, and `dC` are the dimensions of the ``S``, ``T``, ``\\mathcal{T}``, 
and ``C`` marginals. The dimensions of each marginal can be specified manually by setting 
either `dS`, `dT`, `d𝒯`, or `dC` to a *positive* integer number. Alternatively, the dimension
of each marginal can be optimised by setting either `dS`, `dT`, `d𝒯`, or `dC` to an 
instance of [`OptimiseDim`](@ref) 
(e.g. `EmbeddingTE(dT = OptimDim(method_delay = "ac_zero", method_dim = "f1nn")`).

### Specifying delays for generalised delay reconstructions of marginals

The corresponding embedding delay lags are given by `τS`, `τT` and `τC`. The delays
for each marginal can be specified manually by setting either `dS`, `dT`, `d𝒯`, or `dC` 
to a *negative* integer number. The delay defaults for each marginal is -1 (but is set to zero 
if the marginal is one-dimensional), and must always be negative. Alternatively, delays can 
be estimated numerically by setting either `dS`, `dT`, `d𝒯`, and `dC` 
to an instance of [`OptimiseDelay`](@ref) (e.g. `dS = OptimiseDelay(method_delay = "ac_zero")`).

The prediction lag `η` can be either positive or negative, but should not be zero. 

In summary, one can provide

- A single delay ``\\tau``, in which case ``\\tau_{T} = \\{0, \\tau, 2\\tau, \\ldots, (d_{T}- 1)\\tau \\}``, or 
- All the delays manually. If so, then the number of delays must match the dimension of the marginal). 

For the prediction lag, one can provide 

- 
- A single delay ``\\eta_f``, in which case ``\\eta_{\\mathcal{T}} = \\{\\eta_f, 2\\eta_f, \\ldots, (d_{\\mathcal{T}} - 1)\\eta_f \\}``, or 
- All the delays manually. If so, then the number of delays must equal ``d_{\\mathcal{T}}``, which is the dimension of the marginal). 

!!! note
    If both the delay and the dimension for a given marginal is to be estimated numerically, make sure 
    to use the same delay estimation method for both 
    the [`OptimiseDelay`](@ref) and  [`OptimiseDim`](@ref) instances, so that they agree.

## Examples

Default parameters: 

```jldoctest
using CausalityTools
p = EmbeddingTE()

# output
EmbeddingTE(dS=1, dT=1, dC=1, d𝒯=1, τS=-1, τT=-1, τC=-1, η=1)
```

Optimising parameters for the target variable's history (the ``T`` component):

```jldoctest
using CausalityTools
p = EmbeddingTE(
    dT = OptimiseDim(method_delay = "ac_zero", method_dim = "f1nn"),
    τT = OptimiseDelay(method_delay = "ac_zero")
)

# output
EmbeddingTE(dS=1, dT=1, dC=1, d𝒯=1, τS=-1, τT=-1, τC=-1, η=1)
```
"""
@Base.kwdef struct EmbeddingTE
    dS::Union{Int, AbstractVector{Int}, OptimiseDim} = 1
    dT::Union{Int, AbstractVector{Int}, OptimiseDim} = 1
    d𝒯::Union{Int, AbstractVector{Int}, OptimiseDim} = 1
    dC::Union{Int, AbstractVector{Int}, OptimiseDim, Nothing} = 1
    τS::Union{Int, AbstractVector{Int}, OptimiseDelay} = -1
    τT::Union{Int, AbstractVector{Int}, OptimiseDelay} = -1
    η𝒯::Union{Int, AbstractVector{Int}, OptimiseDelay} = 1
    τC::Union{Int, AbstractVector{Int}, OptimiseDelay, Nothing} = -1

    function EmbeddingTE(
            dS::Union{Int, AbstractVector{Int}, OptimiseDim}, 
            dT::Union{Int, AbstractVector{Int}, OptimiseDim}, 
            d𝒯::Union{Int, AbstractVector{Int}, OptimiseDim}, 
            dC::Union{Int, AbstractVector{Int}, OptimiseDim}, 
            τS::Union{Int, AbstractVector{Int}, OptimiseDelay}, 
            τT::Union{Int, AbstractVector{Int}, OptimiseDelay}, 
            η𝒯::Union{Int, AbstractVector{Int}, OptimiseDelay}, 
            τC::Union{Int, AbstractVector{Int}, OptimiseDelay})
        
        if dS isa Int
            dS > 0 || throw(ArgumentError("dimension for marginal S must be a positive integer (got dS=$(dS))"))
        end
        if dT isa Int
            dT > 0 || throw(ArgumentError("dimension for marginal T must be a positive integer (got dT=$(dT))"))
        end
        if dC isa Int
            dC > 0 || throw(ArgumentError("dimension for marginal C must be a positive integer (got dC=$(dC))"))
        end
        if d𝒯 isa Int
            d𝒯 > 0 || throw(ArgumentError("dimension for marginal 𝒯 must be a positive integer (got d𝒯=$(d𝒯))"))
        end
        if τS isa Int
            τS < 0 || throw(ArgumentError("delay for marginal S must be a negative integer (got τS=$(τS))"))
        end
        if τT isa Int
            τT < 0 || throw(ArgumentError("delay for marginal T must be a negative integer (got τT=$(τT))"))
        end
        if τC isa Int
            τC < 0 || throw(ArgumentError("delay for marginal C must be a negative integer (got τC=$(τC))"))
        end
        new(dS, dT, d𝒯, dC, τS, τT, η𝒯, τC)
    end
    
end

function Base.show(io::IO, x::EmbeddingTE)
    s = "EmbeddingTE(dS=$(x.dS), dT=$(x.dT), dC=$(x.dC), d𝒯=$(x.d𝒯), τS=$(x.τS), τT=$(x.τT), τC=$(x.τC), η𝒯=$(x.η𝒯))"
    print(io, s)
end

using CausalityToolsBase
using DelayEmbeddings

"""
    te_embed(source::AbstractVector{T}, target::AbstractVector{T}, 
        p::EmbeddingTE) -> (Vector{Vector}, TEVars, Lags)

Generalised delay reconstruction of `source` and `target` for transfer entropy 
computation using embedding parameters provided by the [`EmbeddingTE`](@ref)
instance `p`.

## Returns

A tuple of 

- the embedding points
- a [`TEVars`](@ref) instance that keeps track of which variables of the embedding belong to 
    which marginals of the reconstruction (indices are: source = 1, target = 2, cond = 3)
- a [`Lags`](@ref) instance, which stores the lags for each variable of the reconstruction.
"""
function te_embed(source::AbstractVector{T}, target::AbstractVector{T}, p::EmbeddingTE) where T
    
    #@show p.τS
    #if (p.τS isa Int && p.τS > 0) || (length(p.τS) > 1 && any(p.τS[p.τS .> 0]))
    #    @warn("Backwards lag τS should be negative. You might be getting nonsensical results!")
    #end
    
    # Get lags and posisions separately for each marginal
    pos_𝒯, lags_𝒯 = rc(target, p.d𝒯, p.η𝒯, true)
    pos_T, lags_T = rc(target, p.dT, p.τT, false)
    pos_S, lags_S = rc(source, p.dS, p.τS, false)
    
    # Add one to the index of the positions for the target (rc doesn't know it is in fact our second time series)
    # TODO: make sure this works when `source` and `target` are multiple time series
    pos_𝒯 .= pos_𝒯 .+ 1
    pos_T .= pos_T .+ 1
    
    pos = Positions([pos_𝒯; pos_T; pos_S])
    lags = Lags([lags_𝒯; lags_T; lags_S])
    
    # TODO: This only works for single time series at the moment
    ts = Dataset(source, target)
    
    # The reconstructed points
    pts = customembed(ts, pos, lags)
    d𝒯 = length(pos_𝒯)
    dT = length(pos_T)
    dS = length(pos_S)

    # Which columns/variables map to which marginals?
    vars = TEVars(
        𝒯  = 1:(d𝒯)           |> collect, 
        T = 1+(d𝒯):dT+(d𝒯)     |> collect, 
        S = 1+(dT+d𝒯):dS+(d𝒯+dT) |> collect)

    return pts, vars, lags
end

"""
    te_embed(source::AbstractVector{T}, target::AbstractVector{T}, cond::AbstractVector{T}, 
        p::EmbeddingTE) -> (Vector{Vector}, TEVars, Lags)

Generalised delay reconstruction of `source`, `target` and `cond` for transfer entropy 
computation using embedding parameters provided by the [`EmbeddingTE`](@ref)
instance `p`.

## Returns

A tuple of 

- the embedding points
- a [`TEVars`](@ref) instance that keeps track of which variables of the embedding belong to 
    which marginals of the reconstruction (indices are: source = 1, target = 2, cond = 3)
- a [`Lags`](@ref) instance, which stores the lags for each variable of the reconstruction.
"""
function te_embed(source::AbstractVector{T}, target::AbstractVector{T}, cond::AbstractVector{T}, p::EmbeddingTE) where T
    
    #@show p.τS
    #if (p.τS isa Int && p.τS > 0) || (length(p.τS) > 1 && any(p.τS[p.τS .> 0]))
    #    @warn("Backwards lag τS should be negative. You might be getting nonsensical results!")
    #end
    # Get lags and posisions separately for each marginal
    pos_𝒯, lags_𝒯 = rc(target, p.d𝒯, p.η𝒯,  true)
    pos_T, lags_T = rc(target, p.dT, p.τT, false)
    pos_S, lags_S = rc(source, p.dS, p.τS, false)
    pos_C, lags_C = rc(cond,   p.dC, p.τC, false)

    # Add one to the index of the positions for the target (rc doesn't know it is in fact our second time series)
    # TODO: make sure this works when `source` and `target` are multiple time series
    pos_𝒯 .= pos_𝒯 .+ 1
    pos_T .= pos_T .+ 1
    pos_C .= pos_C .+ 2

    pos = Positions([pos_𝒯; pos_T; pos_S; pos_C])
    lags = Lags([lags_𝒯; lags_T; lags_S; lags_C])
    
    # TODO: This only works for single time series at the moment
    ts = Dataset(source, target, cond)
    
    # The reconstructed points
    pts = customembed(ts, pos, lags)
    d𝒯 = length(pos_𝒯)
    dT = length(pos_T)
    dS = length(pos_S)
    dC = length(pos_C)

    # Which columns/variables map to which marginals?
    vars = TEVars(
        𝒯 = 1:(d𝒯)               |> collect, 
        T = 1+(d𝒯):dT+(d𝒯)         |> collect, 
        S = 1+(dT+d𝒯):dS+(d𝒯+dT)     |> collect,
        C = 1+(dT+d𝒯+dS):dC+(d𝒯+dT+dS) |> collect)

    return pts, vars, lags
end
