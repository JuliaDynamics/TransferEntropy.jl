export TransferEntropyEstimator

"""
TransferEntropyEstimator

An abstract type for transfer entropy estimators. 
"""
abstract type TransferEntropyEstimator end 

function Base.show(io::IO, estimator::TransferEntropyEstimator)
s = "$(typeof(estimator))($(estimator.b))"
print(io, s)
end


"""
    transferentropy(src, targ, [, cond], emb::EmbeddingTE, method::TransferEntropyEstimator)
    transferentropy(src, targ, [, cond], emb::EmbeddingTE, method::VisitationFrequency)
    transferentropy(src, targ, [, cond], emb::EmbeddingTE, method::NearestNeighborMI)
    transferentropy(src, targ, [, cond], emb::EmbeddingTE, method::SymbolicPerm)
    transferentropy(src, targ, [, cond], emb::EmbeddingTE, method::SymbolicAmplitudeAware)
    transferentropy(src, targ, [, cond], emb::EmbeddingTE, method::TransferOperatorGrid)
    transferentropy(src, targ, [, cond], emb::EmbeddingTE, method::SimplexEstimator)

Estimate transfer entropy[^Schreiber2000] (or, equivalently, conditional mutual information[^Paluš2001]) 
from `src` to `targ`, TE(src → targ), using the provided estimation `method` with delay reconstruction 
parameters `emb`. 

If a third time series `cond` is also provided, compute the conditonal transfer entropy TE(src → targ | cond).

## Arguments 

- **`src`**: The source data series (i.e. enters the `S` part of the generalized embedding; see [`EmbeddingTE`](@ref))
- **`targ`**: The target data series (i.e. enters the `𝒯` and `T` parts of the generalized embedding; see [`EmbeddingTE`](@ref)).
- **`cond`**: An optionally provided data series to condition on (i.e. enters the `C` part of the generalized embedding; see [`EmbeddingTE`](@ref)). 
- **`emb`**: A [`EmbeddingTE`](@ref) instance, containing instructions on how to construct the generalized delay embedding from the input data.
- **`method`**: An instance of a valid transfer entropy estimator, for example [`TransferOperatorGrid`](@ref), [`NearestNeighborMI`](@ref), 
    or [`SymbolicPerm`](@ref).

## Data requirements

No error checking on the inputs is done. Input data must fulfill the following criteria:

- No input time series can contain `NaN` values.
- No input time series can consist of only repeated values of a single point.

## Returns

Returns a single value for the transfer entropy, computed (and summarised, if relevant) according to the `method` specifications.

## Examples

```julia 
x, y, z = rand(100), rand(100), rand(100)

embedding = EmbeddingTE()

# Regular transfer entropy: TE(x → y) in bits, obtained using the 
# `VisitationFrequency` estimator with default coarse-graining settings.
transferentropy(x, y, embedding, VisitationFrequency(b = 2))

# Conditional transfer entropy: TE(x → y | z) in bits, obtained using 
# the `SymbolicPerm` estimator.
transferentropy(x, y, z, embedding, SymbolicPerm(b = 2))
```

[^Schreiber2000]: Schreiber, Thomas. "Measuring information transfer." Physical review letters 85.2 (2000): 461.
[^Paluš2001]: Paluš, M., Komárek, V., Hrnčíř, Z., & Štěrbová, K. (2001). Synchronization as adjustment of information rates: Detection from bivariate time series. Physical Review E, 63(4), 046211.
"""
function transferentropy end

"""
    transferentropy(src, targ, [cond], method::TransferEntropyEstimator; 
            τT::Int=-1, τS::Int=-1, τC::Int=-1, η𝒯::Int=1, 
            dT::Int=1, dS::Int=1, dC::Int=1, d𝒯::Int=1)

Estimate transfer entropy[^Schreiber2000] (or, equivalently, conditional mutual information[^Paluš2001]) 
from `src` to `targ`, TE(src → targ), using the provided estimation `method`. 

`dT`, `dS`, `d𝒯`, and `dC` control the dimension of the marginals; see below. Delay reconstruction lags 
are controlled by `τT`, `τS` and `τC`. `η𝒯` is the prediction lag.  `dC` and `τC` Are ignored if no 
conditional data series are provided.

## Convention for generalized delay reconstruction

This struct contains instructions for transfer entropy computations using the following convention.
Let ``x(t)`` be time series for the source variable, ``y(t)`` be the time series for the target variable and 
``z(t)`` the time series for any conditional variable. To compute transfer entropy, we need the 
following marginals:


```math
\\begin{aligned}
\\mathcal{T}^{(d_{\\mathcal{T}})} &= \\{(y(t+\\eta^{d_{\\mathcal{T}}}), \\ldots, y(t+\\eta^2), y(t+\\eta^1) \\} \\\\
T^{(d_{T})} &= \\{ (y(t+\\tau^0_{T}), y(t+\\tau^1_{T}), y(t+\\tau^2_{T}), \\ldots, y(t + \\tau^{d_{T} - 1}_{T})) \\} \\\\
S^{(d_{S})} &= \\{ (x(t+\\tau^0_{S}), x(t+\\tau^1_{S}), x(t+\\tau^2_{S}), \\ldots, x(t + \\tau^{d_{S} - 1}_{S})) \\} \\\\
C^{(d_{C})} &= \\{ (z(t+\\tau^0_{C}), z(t+\\tau^1_{C}), z(t+\\tau^2_{C}), \\ldots, z(t + \\tau^{d_{C} - 1}_{C})) \\}
\\end{aligned}
```

Depending on the application, the delay reconstruction lags ``\\tau^k_{T} \\leq 0``, ``\\tau^k_{S} \\leq 0``, and ``\\tau^k_{C} \\leq 0`` 
may be equally spaced, or non-equally spaced. The predictions lags ``\\eta^k``may also be equally spaced 
or non-equally spaced, but are always positive. For transfer entropy, convention dictates that at least one 
``\\tau^k_{T}``, one ``\\tau^k_{S}`` and one ``\\tau^k_{C}`` equals zero. This way, the ``T``, ``S`` and ``C`` marginals 
always contains present/past states, 
while the ``\\mathcal T`` marginal contain future states relative to the other marginals. 

Combined, we get the generalized delay reconstruction ``\\mathbb{E} = (\\mathcal{T}^{(d_{\\mathcal{T}})}, T^{(d_{T})}, S^{(d_{S})}, C^{(d_{C})})``. Transfer entropy is then computed as 

```math
\\begin{aligned}
TE_{S \\rightarrow T | C} = \\int_{\\mathbb{E}} P(\\mathcal{T}, T, S, C) \\log_{b}{\\left(\\frac{P(\\mathcal{T} | T, S, C)}{P(\\mathcal{T} | T, C)}\\right)},
\\end{aligned}
```

or, if conditionals are not relevant,

```math
\\begin{aligned}
TE_{S \\rightarrow T} = \\int_{\\mathbb{E}} P(\\mathcal{T}, T, S) \\log_{b}{\\left(\\frac{P(\\mathcal{T} | T, S)}{P(\\mathcal{T} | T)}\\right)},
\\end{aligned}
```

Here, 

- ``\\mathcal{T}`` denotes the ``d_{\\mathcal{T}}``-dimensional set of vectors furnishing the future states of ``T``,
- ``T`` denotes the ``d_{T}``-dimensional set of vectors furnishing the past and present states of ``T``, 
- ``S`` denotes the ``d_{S}``-dimensional set of vectors furnishing the past and present of ``S``, and 
- ``C`` denotes the ``d_{C}``-dimensional set of vectors furnishing the past and present of ``C``.

## Keyword arguments 

### Specifying dimensions for generalized delay reconstructions of marginals

`dS`, `dT`, `d𝒯`, and `dC` are the dimensions of the ``S``, ``T``, ``\\mathcal{T}``, 
and ``C`` marginals. The dimensions of each marginal can be specified manually by setting 
either `dS`, `dT`, `d𝒯`, or `dC` to a *positive* integer number. Alternatively, the dimension
of each marginal can be optimised by setting either `dS`, `dT`, `d𝒯`, or `dC` to an 
instance of [`OptimiseDim`](@ref) 
(e.g. `EmbeddingTE(dT = OptimDim(method_delay = "ac_zero", method_dim = "f1nn")`).

### Specifying delays for generalized delay reconstructions of marginals

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

- A single delay ``\\eta_f``, in which case ``\\eta_{\\mathcal{T}} = \\{\\eta_f, 2\\eta_f, \\ldots, (d_{\\mathcal{T}} - 1)\\eta_f \\}``, or 
- All the delays manually. If so, then the number of delays must equal ``d_{\\mathcal{T}}``, which is the dimension of the marginal). 

!!! note
    If both the delay and the dimension for a given marginal is to be estimated numerically, make sure 
    to use the same delay estimation method for both 
    the [`OptimiseDelay`](@ref) and  [`OptimiseDim`](@ref) instances.
"""
function transferentropy(src, targ, method::TransferEntropyEstimator; 
        τT = -1, τS  = -1, τC = -1, η𝒯 = 1, 
        dT = 1, dS = 1, dC = 1, d𝒯 = 1)
    
    emb = EmbeddingTE(dT = dT, dS = dS, dC = dC, d𝒯 = d𝒯,
                      τT = τT, τS  = τS, τC = τC, η𝒯 = η𝒯)
    
    transferentropy(src, targ, emb, method)
end

function transferentropy(src, targ, cond, method::TransferEntropyEstimator; 
    τT = -1, τS  = -1, τC = -1, η𝒯 = 1, dT = 1, dS = 1, dC = 1, d𝒯 = 1)

    emb = EmbeddingTE(dT = dT, dS = dS, dC = dC, d𝒯 = d𝒯,
                    τT = τT, τS  = τS, τC = τC, η𝒯 = η𝒯)

    transferentropy(src, targ, cond, emb, method)
end

# Low-level method
function _transferentropy end
