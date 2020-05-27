"""
    transferentropy(source, target, embedding::EmbeddingTE, estimator::TransferEntropyEstimator)
    transferentropy(source, target, cond, embedding::EmbeddingTE, estimator::TransferEntropyEstimator)

Compute transfer entropy from `source` to `target` (conditioned on `cond` if given). 
The computation is done on a generalized delay reconstruction constructed from the input time series using 
the parameters in `embedding`, using the provided `estimator`.

## Arguments 

- **`source`**: The source data series (i.e. enters the `S` part of the generalized embedding)
- **`target`**: The target data series (i.e. enters the `𝒯` and `T` parts of the generalized embedding).
- **`cond`**: The data series to condition on (i.e. enters the `C` part of the generalized embedding). For 
    bivariate analyses, do not provide this argument.
- **`embedding`**: Instructions for how to construct the generalized delay embedding from the input time 
    series, given as a [`EmbeddingTE`](@ref) instance.
- **`estimator`**: A valid transfer entropy estimator. Currently available choices are [`VisitationFrequency`](@ref)
    and [`TransferOperatorGrid`](@ref).

## Returns 

Returns a single value for the transfer entropy, computed and summarised according to the `estimator` specifications.

## Data requirements

No error checking on the data is done. Input data must fulfill the following criteria:

- No input time series can consist of a single point.
- No input time series can contain `NaN` values.

## Examples

```julia 
x, y, z = rand(100), rand(100), rand(100)

est_vf = VisitationFrequency()
embedding = EmbeddingTE()

# Regular transfer entropy
transferentropy(x, y, embedding, est_vf)

# Conditional transfer entropy
transferentropy(x, y, z, embedding, est_vf)
```

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