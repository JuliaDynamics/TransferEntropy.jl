import Entropies: SymbolicPermutation, CountOccurrences, symbolize, symbolize!
export SymbolicPermutation

function transferentropy(s, t, est::SymbolicPermutation; 
        base = 2, α = 1, τT = -1, τS = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1)

    # Generalized embeddings of each marginal separately, using the same dimension for each
    # marginal. Then symbolize the embedded time series, which gives integer symbol sequences.
    symb_s = symbolize(s, est)
    symb_t = symbolize(t, est)
    
    # After symbolization, use the general interface with occurrence frequency entropy 
    # estimator (probabilities are obtained from histograms of the symbols).
    transferentropy(symb_s, symb_t, CountOccurrences(), 
        τT = τT, τS = τS, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯)
end

function transferentropy!(symb_s, symb_t, s, t, est::SymbolicPermutation; base = 2, α = 1, 
        τT = -1, τS = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1)

    symbolize!(symb_s, s, est)
    symbolize!(symb_t, t, est)
    
    transferentropy(symb_s, symb_t, CountOccurrences(), 
        τT = τT, τS = τS, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯)
end

function transferentropy(s, t, c, est::SymbolicPermutation; base = 2, α = 1, 
        τT = -1, τS = -1, τC = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1, dC = 1)

    symb_s = symbolize(s, est)
    symb_t = symbolize(t, est)
    symb_c = symbolize(c, est)

    transferentropy(symb_s, symb_t, symb_c, CountOccurrences(),
        τT = τT, τS = τS, τC = τC, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯, dC = dC)
end

function transferentropy!(symb_s, symb_t, symb_c, s, t, c, est::SymbolicPermutation; 
        base = 2, α = 1, 
        τT = -1, τS = -1, τC = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1, dC = 1)

    symbolize!(symb_s, s, est)
    symbolize!(symb_t, t, est)
    symbolize!(symb_c, c, est)
    
    transferentropy(symb_s, symb_t, symb_c, CountOccurrences(), 
        τT = τT, τS = τS, τC = τC, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯, dC = dC)
end