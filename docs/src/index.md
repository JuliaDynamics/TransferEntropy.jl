# TransferEntropy.jl

This package exports two functions, [`transferentropy`](@ref) and [`mutualinfo`](@ref).
Both functions use the estimators listed below (not all estimators are implemented for both 
functions; see docstrings for [`transferentropy`](@ref) and [`mutualinfo`](@ref) for 
details).

## Estimators

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

## Transfer entropy

```@docs
transferentropy
```

## Mutual information

```@docs
mutualinfo
```

## Dataset

```@docs
Dataset
```