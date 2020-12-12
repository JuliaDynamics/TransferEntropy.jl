# TransferEntropy.jl

## Exported functions

This package exports two functions, [`transferentropy`](@ref) and [`mutualinfo`](@ref).

In order to compute either quantity, combine your input data with one of the available 
[estimators](@ref estimators). Docstrings for [`transferentropy`](@ref) and 
[`mutualinfo`](@ref) give overviews of currently implemented estimators for either 
function.

## [Estimators](@id estimators)

### Binning based

```@docs
VisitationFrequency
RectangularBinning
```

### Kernel density based

```@docs
NaiveKernel
TreeDistance
DirectDistance
```

### Nearest neighbor based

```@docs
KozachenkoLeonenko
Kraskov
Kraskov1
Kraskov2
```

### Permutation based

```@docs
SymbolicPermutation
```

### Hilbert

```@docs
Hilbert
Amplitude
Phase
```
