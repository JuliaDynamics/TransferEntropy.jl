# TransferEntropy.jl

Julia package for computing transfer entropy (TE), which is part of the   [`CausalityTools.jl`](https://github.com/kahaaga/CausalityTools.jl) package
 for the detection of causal relationships and the computation of invariant
 measures from time series.

Three estimators are currently estimated.

1. A k nearest neighbours (kNN) estimator (`transferentropy_kraskov`, or the alias `tekraskov`). This estimator uses the Kraskov estimator for mutual information to compute TE.
2. A visitation frequency based estimator (`transferentropy_visitfreq`, or the alias `tefreq`). This estimator is based on state space discretization, and computes a probability distribution over the bins as frequency at which the orbit of the system visits the bins.
3. A transfer operator based estimator (`transferentropy_transferoperator_visitfreq`, or the alias `tetofreq`). This is a new estimator from an upcoming paper by Diego, Haaga and Hannisdal. It is also binning based, but estimates TE from the invariant
distribution arising from an approximation of the transfer operator on the state
space reconstruction.

## Installation
`TransferEntropy.jl` relies on the `PerronFrobenius.jl` package, which is not
yet on METADATA. You may install both packages by running the following
lines in the Julia console

```julia
Pkg.clone("https://github.com/kahaaga/TransferEntropy.jl")
Pkg.clone("https://github.com/kahaaga/PerronFrobenius.jl")
```
