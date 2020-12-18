# Renyi entropies are not defined for nearest neighbor estimators, so need an extra method
# that doesn't feed the alpha keyword to genentropy
function transferentropy(joint, ST, T𝒯, T, est::NearestNeighborEntropyEstimator; 
        base = 2, q = 1)
    
    te = genentropy(T𝒯, est, base = base) +
        genentropy(ST, est, base = base) -
        genentropy(T, est, base = base) -
        genentropy(joint, est, base = base)
end