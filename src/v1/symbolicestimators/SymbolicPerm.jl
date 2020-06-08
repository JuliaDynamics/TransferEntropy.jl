export SymbolicPerm

import DelayEmbeddings: Dataset
import Combinatorics: nthperm

"""
    SymbolicPerm(; b::Real = 2, m::Int = 3) <: SymbolicTransferEntropyEstimator

A permutation-based symbolic transfer entropy estimator [^Staniek2008]. 

Computes probabilities with the logarithm to base `b` (the default,`b = 2`, 
gives the transfer entropy in bits).

## Estimation steps

- Delay reconstruct the input time series into relevant marginals. This results in 
    the marginal delay reconstructions `𝒯`, `T` and `S` (and `C` if conditional analyses are to 
    be performed; see see [`EmbeddingTE`](@ref)). Each marginal has dimension `m`, but may
    be constructed with different prediction lag(s) `η𝒯` and embedding lag(s) `τT`, `τS` (and `τC`).
- Permutation symbolize the delay vectors in `𝒯` (see Stanieck and Lehnertz, 2008[^Staniek2008]). 
    Because each marginal is `m`-dimensional, delay vectors are mapped to `m`-element 
    permutations. Each permutation is then onto a unique integer symbol. 
- The marginal delay reconstructions are now one-dimensional, and consist of symbol (integer) sequences,
    which can be regarded as regular time series from which we can estimate transfer entropy.
- Estimate probabilities and marginal entropies from relative frequencies of symbol occurrences.

## Implementation details 

The original symbolic transfer entropy (STE) implementation seems to use the same delay embedding lag 
for the target time series for both the `𝒯`the `T` marginals, by time-shifting the target time series *after* it has 
been symbolized. In this implementation, the delay reconstructions for the marginals `𝒯` and `T` are 
done separately (with possibly different embedding lags) *before* symbolization. Thus, when 
computing transfer entropy, `𝒯` and `T` are actually different symbol sequences, not 
time-shifted but otherwise equal symbol sequences. 

Another difference from the original implementation is that they use the same constant 
embedding lag for both the `T` and `S` marginals, while here you may use different embedding 
lags for different marginals (see [`EmbeddingTE`](@ref)).

Together, these differences put our implemention closer to the "transfer entropy on rank vectors" (TERV)
from Kugiumtzis (2010)[^Kugiumtzis2010], which is conceptually similar, but allows different 
embedding dimensions for the different marginals.

When applied to the same unidirectional example systems as in the original paper[^Staniek2008], these differences yield 
qualitatively similar preferred directions of information flow, although differing in absolute magnitude.

[^Staniek2008]: Staniek, Matthäus, and Klaus Lehnertz. "Symbolic transfer entropy." Physical Review Letters 100.15 (2008): 158101.
[^Kugiumtzis2010]: Kugiumtzis, Dimitris. "Transfer entropy on rank vectors." arXiv preprint arXiv:1007.0357 (2010).
"""
struct SymbolicPerm <: SymbolicTransferEntropyEstimator
    b::Real
    m::Int 
    
    function SymbolicPerm(; b::Real = 2, m::Int = 3)
        m >= 2 || throw(ArgumentError("Dimensions of individual marginals must be at least 2. Otherwise, symbol sequences cannot be assigned to the marginals. Got m=$(m)."))

        new(b, m)
    end
end

function symbolize(E::Dataset, method::SymbolicPerm)
    m = length(E[1])
    n = length(E)

    # Pre-allocate symbol vector
    symb = zeros(Int, n)
    
    #= 
    Loop over embedding vectors `E[i]`, find the indices `p_i` that sort each `E[i]`,
    then get the corresponding integers `k_i` that generated the 
    permutations `p_i`. Those integers are the symbols for the embedding vectors
    `E[i]`.
    =#
    sp = zeros(Int, m)
    @inbounds for i = 1:n
        sortperm!(sp, E[i])
        symb[i] = nthperm(sp)
    end
    
    return symb
end