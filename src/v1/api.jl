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

# Low-level method
function _transferentropy end

"""
    TransferEntropyEstimator

An abstract type for transfer entropy estimators. 
"""
abstract type TransferEntropyEstimator end 

function Base.show(io::IO, estimator::TransferEntropyEstimator)
    s = "$(typeof(estimator))($(estimator.b))"
    print(io, s)
end

export TransferEntropyEstimator