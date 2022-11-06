import Entropies: SymbolicPermutation, CountOccurrences, outcomes, outcomes!
export SymbolicPermutation

# For two or three timeseries as inputs. Each input timeseries is converted to an
# `s ∈ (1, 2, ..., m!)`
function transferentropy(e::Entropy, est::SymbolicPermutation, args...)
    # Generalized embeddings of each marginal separately, using the same dimension for each
    # marginal.
    encoding = OrdinalPattern(m = est.m, τ = est.τ, lt = est.lt)

    # Encode embedding vectors to symbol sequences. `symbols[k][j] ∈ (1, 2, ..., m!)`
    # is the symbol for the `j`-th vector in the embedding for the `k`-th input timeseries.
    symbols = outcomes.([args...,], encoding)

    # Compute transfer entropy from symbol vectors, using sum of marginal entropies,
    # and just counting frequencies of symbols.
    transferentropy(e, CountOccurrences(), symbols...)
end

function transferentropy!(πs, πt, e::Entropy, est::SymbolicPermutation, s, t)
    encoding = OrdinalPatternEncoding(m = est.m, τ = est.τ, lt = est.lt)
    for (ts, πts) in zip((s, t), (πs, πs))
        outcomes!(πts, ts, encoding)
    end
    return transferentropy(e, CountOccurrences(), πs, πt, πc)
end

function transferentropy!(πs, πt, πc, e::Entropy, s, t, c,
        est::SymbolicPermutation; kwargs...)
    encoding = OrdinalPatternEncoding(m = est.m, τ = est.τ, lt = est.lt)
    for (ts, πts) in zip((s, t, c), (πs, πt, πc))
        outcomes!(πts, ts, encoding)
    end
    return transferentropy(e, CountOccurrences(), πs, πt, πc)
end

transferentropy(est::SymbolicPermutation, args...) =
        transferentropy(Shannon(; base), est, args...)
transferentropy!(πs, πt, est::SymbolicPermutation, s, t) =
    transferentropy!(πs, πt, Shannon(; base), est, s, t)
transferentropy!(πs, πt, πc, est::SymbolicPermutation, s, t, c) =
    transferentropy!(πs, πt, πc, Shannon(; base), est, s, t, c)
