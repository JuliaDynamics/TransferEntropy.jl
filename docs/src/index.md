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
TransferOperator
RectangularBinning
```

### Kernel density based

```@docs
NaiveKernel
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

There is a possible performance optimization to be made with this method:
```julia
transferentropy!(symb_s, symb_t, s, t, [c], est::SymbolicPermutation; 
        base = 2, q = 1, m::Int = 3, τ::Int = 1, ...) → Float64
```
You can optionally provide pre-allocated (integer) symbol vectors `symb_s` and `symb_t` (and `symb_c`),
where `length(symb_s) == length(symb_t) == length(symb_c) == N - (est.m-1)*est.τ`. This is useful for saving 
memory allocations for repeated computations.

### Hilbert

```@docs
Hilbert
Amplitude
Phase
```

## Automated variable selection

```@docs
BBNUE
```