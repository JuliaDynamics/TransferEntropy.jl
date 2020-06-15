
import CausalityToolsBase: non0hist

import ..transferentropy
import ..EmbeddingTE 
import ..te_embed

export transferentropy

""" 
    encode_pattern(motif)

Encode the `motif` (a vector of indices that would sort some embedding vector `v` in ascending order) 
into its unique integer symbol according to Algorithm 1 in Berger et al. (2019)[^Berger2019].

[Berger2019]: Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.
"""
function encode_pattern(x, m::Int = length(x))
    n = 0
    for i = 1:m-1
        for j = i+1:m
            n += x[i] > x[j] ? 1 : 0
        end
        n = (m-i)*n
    end
    
    return n
end

function transferentropy(source, target, embedding::EmbeddingTE, method::SymbolicTransferEntropyEstimator)
    m = method.m

    # Generalized embeddings of each marginal separately.
    # Use provided delays, but overwrite dimensions with `m`
    # (all marginals must have dimension `m` to have symbols 
    # of length `m`)
    symbolemb = EmbeddingTE(dT = m, 
                            dS = m, 
                            dC = m, 
                            d𝒯 = m, 
                            τT = embedding.τT,
                            τS = embedding.τS,
                            τC = embedding.τC,
                            η𝒯 = embedding.η𝒯)
    
    pts, vars, τs, js  = te_embed(source, target, symbolemb)

    E𝒯 = pts[:, 1:m]       
    ET = pts[:, m+1:2*m]   
    ES = pts[:, 2*m+1:end] 
    
    # Symbolize each marginal separately
    s𝒯 = symbolize(E𝒯, method)
    sT = symbolize(ET, method)
    sS = symbolize(ES, method)
    
    # Treat each symbolized marginal as a time series and compute 
    # transfer entropy from the entropies of the marginal histograms
    ST = non0hist(Dataset(sS, sT).data)
    T𝒯 = non0hist(Dataset(sT, s𝒯).data)
    T = non0hist(Dataset(sT).data)
    joint = non0hist(Dataset(s𝒯, sT, sS).data)
    
    te = StatsBase.entropy(ST, method.b) +
         StatsBase.entropy(T𝒯, method.b) -
         StatsBase.entropy(T, method.b) -
         StatsBase.entropy(joint, method.b)
end

function transferentropy(source, target, cond, embedding::EmbeddingTE, method::SymbolicTransferEntropyEstimator)
    m = method.m

    # Generalized embeddings of each marginal separately.
    # Use provided delays, but overwrite dimensions with `m`
    # (all marginals must have dimension `m` to have symbols 
    # of length `m`)
    symbolemb = EmbeddingTE(dT = m, 
                            dS = m, 
                            dC = m, 
                            d𝒯 = m, 
                            τT = embedding.τT,
                            τS = embedding.τS,
                            τC = embedding.τC,
                            η𝒯 = embedding.η𝒯)
    pts, vars, τs, js  = te_embed(source, target, cond, symbolemb)

    E𝒯 = pts[:, 1:m]       
    ET = pts[:, m+1:2*m]   
    ES = pts[:, 2*m+1:3*m] 
    EC = pts[:, 3*m+1:end] 
    
    # Symbolize each marginal separately
    s𝒯 = symbolize(E𝒯, method)
    sT = symbolize(ET, method)
    sS = symbolize(ES, method)
    sC = symbolize(EC, method)

    # Treat each symbolized marginal as a time series and compute 
    # transfer entropy from relative occurrences 
    symb = Dataset(s𝒯, sT, sS, sC)

    # Compute transfer entropy as sum of entropies of the 
    # different marginals
    ST = non0hist(Dataset(sS, sT, sC).data)
    T𝒯 = non0hist(Dataset(sT, s𝒯, sC).data)
    T = non0hist(Dataset(sT, sC).data)
    joint = non0hist(Dataset(s𝒯, sT, sS, sC).data)
    
    te = StatsBase.entropy(ST, method.b) +
         StatsBase.entropy(T𝒯, method.b) -
         StatsBase.entropy(T, method.b) -
         StatsBase.entropy(joint, method.b)
end