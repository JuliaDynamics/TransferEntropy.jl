function transferentropy(source, target, est::AmplitudeAndPermutation; base = 2, q = 1,
        τT = -1, τS = -1, τC = -1, η𝒯 = 1, dT = 1, dS = 1, d𝒯 = 1, dC = 1)

    emb =  TransferEntropy.EmbeddingTE(τT = τT, τS = τS, η𝒯 = η𝒯, dT = dT, dS = dS, d𝒯 = d𝒯)

    # Generalized embeddings of each marginal separately.
    # Use provided delays, but overwrite dimensions with `m`
    # (all marginals must have dimension `m` to have symbols
    # of length `m`)
    m = est.m
    symbolemb =  TransferEntropy.EmbeddingTE(dT = m, dS = m, dC = m, d𝒯 = m,
        τT = emb.τT, τS = emb.τS, τC = emb.τC, η𝒯 = emb.η𝒯)

    pts, vars, τs, js = TransferEntropy.te_embed(source, target, symbolemb)

    # Separate marginals
    E𝒯 = pts[:, 1:m]
    ET = pts[:, m+1:2*m]
    ES = pts[:, 2*m+1:end]

    # Symbolized marginals
    s𝒯 = symbolize(E𝒯, est)
    sT = symbolize(ET, est)
    sS = symbolize(ES, est)

    pST = Entropies.probabilities(Dataset(sS, sT))
    pT𝒯 = Entropies.probabilities(Dataset(sT, s𝒯))
    pT =  Entropies.probabilities(Dataset(sT))
    pjoint = Entropies.probabilities(Dataset(s𝒯, sT, sS))

    return Entropies.genentropy(pT𝒯, base = base, q = q) +
        Entropies.genentropy(pST, base = base, q = q) -
        Entropies.genentropy(pT, base = base, q = q) -
        Entropies.genentropy(pjoint, base = base, q = q)
end

x = rand(10000)
y = rand(10000)
te = transferentropy(x, y, AmplitudeAndPermutation())
