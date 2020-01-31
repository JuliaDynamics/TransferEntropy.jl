include("low_level_estimation/low_level.jl")

"""
    transferentropy(source, target, embedding::EmbeddingTE, estimator::TransferEntropyEstimator)
    transferentropy(source, target, cond, embedding::EmbeddingTE, estimator::TransferEntropyEstimator)

Compute transfer entropy from `source` to `target` (conditioned on `cond` if given). 
The computation is done on a generalised delay reconstruction constructed from the input time series using 
the parameters in `embedding`, using the provided `estimator`.

## Arguments 

- **`source`**: The source data series (i.e. enters the `S` part of the generalised embedding)
- **`target`**: The target data series (i.e. enters the `𝒯` and `T` parts of the generalised embedding).
- **`cond`**: The data series to condition on (i.e. enters the `C` part of the generalised embedding). For 
    bivariate analyses, do not provide this argument.
- **`embedding`**: Instructions for how to construct the generalised delay embedding from the input time 
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


#####################
# Binning estimators
#####################


function transferentropy(source, target, embedding::EmbeddingTE, estimator::Estimators.BinningTransferEntropyEstimator)
    # Generalised delay embedding
    pts, vars, lags = te_embed(source, target, embedding)

    # Get the binning (if a heuristic is used, determine binning from input time series and dimension)
    if estimator.binning isa BinningHeuristic
        total_dim = length(pts[1])
        binning = Estimators.estimate_partition(target, total_dim, estimator.binning)
    else
        binning = estimator.binning
    end

    # Compute TE over different partitions
    bs = binning isa Vector{RectangularBinning} ? binning : [binning]
    tes = map(binscheme -> _transferentropy(pts, vars, binscheme, estimator), bs)

    return estimator.summary_statistic(tes)
end

function transferentropy(source, target, cond, embedding::EmbeddingTE, estimator::Estimators.BinningTransferEntropyEstimator)
    # Generalised delay embedding
    pts, vars, lags = te_embed(source, target, cond, embedding)

    # Get the binning (if a heuristic is used, determine binning from input time series and dimension)
    if estimator.binning isa BinningHeuristic
        total_dim = length(pts[1])
        binning = Estimators.estimate_partition(target, total_dim, estimator.binning)
    else
        binning = estimator.binning
    end

    # Compute TE over different partitions
    bs = binning isa Vector{RectangularBinning} ? binning : [binning]
    tes = map(binscheme -> _transferentropy(pts, vars, binscheme, estimator), bs)

    return estimator.summary_statistic(tes)
end

