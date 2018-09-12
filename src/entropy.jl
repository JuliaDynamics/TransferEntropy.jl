"""
	entropy(probdist, b = 2) -> Float64

Compute the entropy (to the base `b`) of a probability distribution.
"""
function entropy(probdist; b = 2)
    te = 0.0
    @inbounds for i = 1:size(probdist, 1)
        te -= probdist[i] * log(b, probdist[i])
    end
    return te
end

"""
	entropy(probdist, b) -> Float64

Compute the entropy (to the base `b`) of a probability distribution.
"""
function entropy(probdist, b)
    te = 0.0
    @inbounds for i = 1:size(probdist, 1)
        te -= probdist[i] * log(b, probdist[i])
    end
    return te
end
