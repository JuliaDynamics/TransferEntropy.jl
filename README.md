# TransferEntropy.jl

[![Build Status](https://travis-ci.org/kahaaga/TransferEntropy.jl.svg?branch=master)](https://travis-ci.org/kahaaga/TransferEntropy.jl)

Julia package for computing transfer entropy (TE), conditional mutual information (CMI) or any other information theoretic functional.

This package provides essential algorithms for the [`CausalityTools.jl`](https://github.com/kahaaga/CausalityTools.jl) package, which provides methods to detect causal relationship from time series, and tools for computating the transfer operator and invariant measures from time series.

## Transfer entropy estimators
Currently, the following three estimators are implemented and tested. For details on

| Estimator (and aliases) | Accepts  | Details | Reference  |
|---|---|---|---|
| `transferentropy_transferoperator_grid` (`tetogrid`) | `AbstractArray`, `AbstractEmbedding`  | A new estimator that computes tranfer entropy from an invariant measure of an approximation to the transfer operator. The transfer operator is approximated using the  `transferoperator_grid` estimator from [PerronFrobenius.jl](https://github.com/kahaaga/PerronFrobenius.jl)| [Diego et al. (2018)](https://arxiv.org/abs/1811.01677) |
| `transferentropy_visitfreq` (`tefreq`)   | `AbstractArray`, `AbstractEmbedding` | A classic, naive binning-based transfer entropy estimator. Obtains the probability distribution from the frequencies at which the orbit visits the different regions of the reconstructed attractor  | [Diego et al. (2018)](https://arxiv.org/abs/1811.01677)|
| `transferentropy_kraskov` (`tekraskov`, `tekNN`) | `AbstractArray`, `AbstractEmbedding`  | A k Nearest Neigbours (kNN) transfer entropy estimator. Computes the transfer entropy as the sum of two mutual information (MI) terms, which are computed using the Kraskov MI estimator | [Diego et al. (2018)](https://arxiv.org/abs/1811.01677), [Kraskov et al. (2004)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138) |


## Installation
Run the following lines in the Julia console to install the package.

```julia
using Pkg
Pkg.add("TransferEntropy")
```
