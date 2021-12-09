using TimeseriesSurrogates

export bbnue

# Keep this for when a common interface for optimized variable selection methods has been established
# """
#export BBNUE
#     BBNUE(est) <: TransferEntropyEstimator

# The bootstrap-based non-uniform embedding estimator (BB-NUE) for conditional transfer entropy (Montalto et al., 2014).
# Uses the estimator `est` to compute relevant marginal entropies. (e.g. `VisitationFrequency(RectangularBinning(3))`) 

# [^Montalto2014]: Montalto, A.; Faes, L.; Marinazzo, D. MuTE: A MATLAB toolbox to compare established and novel estimators of the multivariate transfer entropy. PLoS ONE 2014, 9, e109462.
# """
# struct BBNUE{E} <: TransferEntropyEstimator
#     est::E
# end


""" 
    bbnue(source, target, [cond], est::BBNUE; q = 0.95, η = 1, 
        nsurr = 100, uq = 0.95, 
        include_instantaneous = true, 
        method_delay = "ac_min", 
        maxlag::Union{Int, Float64} = 0.05
        ) → te, js, τs, idxs_source, idxs_target, idxs_cond

Estimate transfer entropy using the bootstrap-based non-uniform embedding (BBNUE) estimator, 
which uses a bootstrap-basedcriterion to identify the most relevant and minimally redundant 
variables from the present/past of `source`, present/past `cond` (if given) and the past of 
`target` that contribute most to `target`'s future. `η` is the forward prediction lag. 
Multivariate `source`, `target` and `cond` (if given) are all possible.

For significance testing of a variable, `nsurr` circular shift surrogates are generated, 
and if transfer entropy for the original variables exceeds the `uq`-quantile of that of the 
surrogate ensemble, then the variable is included.

If `instantaneous` is `true`, then instantaneous interactions are also considered, i.e. effects like 
`source(t) → target(t)` are allowed.

In this implementation, the maximum lag for each embedding variable is determined using `estimate_delay` 
from `DelayEmbeddings`. The keywords `method_delay` (default is "ac_min") controls the method 
for estimating the delay, and `maxlag` is the maximum allowed delay (if `maxlag ∈ [0, 1]` is a fraction, 
then the maximum lag is that fraction of the input time series length, and if `maxlag` is an integer, 
then the maximum lag is `maxlag`).

## Implementation details

Currently, only this implementation is optimized for the bin-estimator approach from 
Montalto et al. (2014)[^Montalto2014], which uses a conditional entropy minimization criterion for 
select variables.  Their bin-estimator approach corresponds to using 
the `VisitationFrequency` estimator with bins whose sides are equal-length, e.g. 
`VisitationFrequency(RectangularBinning(0.5))`. Here, you can use any desired rectangular binning.

It is also possible to use other entropy estimators than `VisitationFrequency` to compute entropies, 
but this implementation will use conditional entropy minimization regardless of the choice of 
estimator.

## Returns

A 6-tuple is returned, consisting of:
- `te`: The computed transfer entropy value. If no relevant variables were selected, then `te = 0.0`.
- `js`: The indices of the selected variables. `js[i]` is the `i`-th entry in the array `[idxs_source..., idxs_target..., idxs_cond...,]`.
- `τs`: The embedding lags of the selected variables. `τs[i]` corresponds to `js[i]`.
- `idxs_source`: The indices of the source variables.
- `idxs_target`: The indices of the target variables.
- `idxs_cond`: The indices of the conditional variables (empty if `cond` is not given).

## Example

```julia
using CausalityTools, DynamicalSystems
sys = ExampleSystems.logistic2_unidir(c_xy = 1.5)
orbit = trajectory(sys, 10000, Ttr = 10000)
x, y = columns(orbit)

# Use a coarse-grained rectangular binning with subdivisions in each dimension,
# to keep computation costs low and to ensure the probability distributions 
# over the bins don't approach the uniform distribution (need enough points 
# to fill bins).
est = VisitationFrequency(RectangularBinning(3))
te_xy, params_xy = bbnue(x, y, BBNUE(est))
te_yx, params_yx = bbnue(y, x, BBNUE(est))

te_xy, te_yx
```

[^Montalto2014]: Montalto, A.; Faes, L.; Marinazzo, D. MuTE: A MATLAB toolbox to compare established and novel estimators of the multivariate transfer entropy. PLoS ONE 2014, 9, e109462.
"""
function bbnue(source, target, cond, est; q = 0.95, η = 1, nsurr = 100, uq = 0.95, 
        include_instantaneous = true, method_delay = "ac_min", maxlag::Union{Int, Float64} = 0.05)

    Ω, Y⁺, τs, js, idxs_source, idxs_target, idxs_cond = 
        embed_candidate_variables(
            process_input(source), 
            process_input(target), 
            process_input(cond), 
            η = η)

    return optim_te(Ω, Y⁺, τs, js, idxs_source, idxs_target, idxs_cond, est, q = q, nsurr = 19, uq = uq)
end

function bbnue(source, target, est; q = 0.95, η = 1, nsurr = 100, uq = 0.95, 
        include_instantaneous = true, method_delay = "ac_min", maxlag::Union{Int, Float64} = 0.05)

    Ω, Y⁺, τs, js, idxs_source, idxs_target, idxs_cond = 
        embed_candidate_variables(process_input(source), process_input(target), η = η)

    return optim_te(Ω, Y⁺, τs, js, idxs_source, idxs_target, idxs_cond, est, q = q, nsurr = 19, uq = uq)
end