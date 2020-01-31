include("low_level_estimation/low_level.jl")

"""
    transferentropy(source, target, embedding::EmbeddingTE, estimator::TransferEntropyEstimator)

Compute transfer entropy from `source` to `target`. The computation is done on a generalised delay reconstruction  
constructed from the input time series using the parameters in `embedding`, using the provided `estimator`.

## Data requirements:
- No input time series can consist of a single point.
- No input time series can contain `NaN` values.
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

