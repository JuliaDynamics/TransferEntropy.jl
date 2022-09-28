import Entropies: SymbolicPermutation, CountOccurrences, symbolize, symbolize!
export SymbolicPermutation

function transferentropy(e::Entropy, s, t, est::SymbolicPermutation; kwargs...)

    # Generalized embeddings of each marginal separately, using the same dimension for each
    # marginal. Then symbolize the embedded time series, which gives integer symbol sequences.
    s_est = OrdinalPattern(m = est.m, τ = est.τ, lt = est.lt)
    symb_s = symbolize(s, s_est)
    symb_t = symbolize(t, s_est)

    # After symbolization, use the general interface with occurrence frequency entropy
    # estimator (probabilities are obtained from histograms of the symbols).
    transferentropy(e, symb_s, symb_t, CountOccurrences(); kwargs...)
end

function transferentropy!(symb_s, symb_t, s, t, est::SymbolicPermutation; kwargs...)
    s_est = OrdinalPattern(m = est.m, τ = est.τ, lt = est.lt)
    symbolize!(symb_s, s, s_est)
    symbolize!(symb_t, t, s_est)

    transferentropy(symb_s, symb_t, CountOccurrences(); kwargs...)
end

function transferentropy(e::Entropy, s, t, c, est::SymbolicPermutation; base = 2,
        kwargs...)

    s_est = OrdinalPattern(m = est.m, τ = est.τ, lt = est.lt)
    symb_s = symbolize(s, s_est)
    symb_t = symbolize(t, s_est)
    symb_c = symbolize(c, s_est)

    transferentropy(e, symb_s, symb_t, symb_c, CountOccurrences(); kwargs...)
end

function transferentropy!(e::Entropy, symb_s, symb_t, symb_c, s, t, c,
        est::SymbolicPermutation; kwargs...)

    s_est = OrdinalPattern(m = est.m, τ = est.τ, lt = est.lt)
    symbolize!(symb_s, s, s_est)
    symbolize!(symb_t, t, s_est)
    symbolize!(symb_c, c, s_est)

    transferentropy(e, symb_s, symb_t, symb_c, CountOccurrences(); kwargs...)
end

transferentropy(s, t, est::SymbolicPermutation; base = 2, kwargs...) =
        transferentropy(Shannon(; base), s, t, est; kwargs...)
transferentropy(s, t, c, est::SymbolicPermutation; base = 2, kwargs...) =
    transferentropy(Shannon(; base), s, t, c, est; kwargs...)
transferentropy!(symb_s, symb_t, e::Entropy, s, t, est::SymbolicPermutation; base = 2,
        kwargs...) =
    transferentropy!(symb_s, symb_t, Shannon(; base), s, t, est; kwargs...)
transferentropy!(symb_s, symb_t, symb_c, s, t, c, est::SymbolicPermutation;
    base = 2, kwargs...) =
    transferentropy!(Shannon(; base), symb_s, symb_t, symb_c, s, t, c, est)
