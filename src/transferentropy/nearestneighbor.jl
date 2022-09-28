using Entropies: Entropy, Renyi


# Transfer entropy is not defined for all types of entropies.
function _transferentropy(e::E, joint, ST, T𝒯, T, est::SimpleNNEstimator) where E <: Entropy
    throw(ArgumentError("$(E) transfer entropy not defined for $(E) estimator"))
end

# For Shannon it is. Specialize on other entropy types too if applicable.
function _transferentropy(e::Renyi, joint, ST, T𝒯, T, est::SimpleNNEstimator)
    te = entropy(e, T𝒯, est) +
        entropy(e, ST, est) -
        entropy(e, T, est) -
        entropy(e, joint, est)
end

_transferentropy(joint, ST, T𝒯, T, est::SimpleNNEstimator; base = 2) =
    _transferentropy(Shannon(; base), joint, ST, T𝒯, T)
