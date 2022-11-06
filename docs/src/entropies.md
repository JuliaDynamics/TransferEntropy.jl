
# Entropies

## Types of entropies

```@autodocs
Modules = [Entropies]
Filter = e -> e ∈ subtypes(Entropy)
```

## Dedicated estimators

Here we list functions which compute Shannon entropies via alternate means, without explicitly computing some probability distributions and then using the Shannon formula.

```@docs
IndirectEntropy
```

### Nearest neighbor based

```@docs
Kraskov
KozachenkoLeonenko
Zhu
ZhuSingh
```
