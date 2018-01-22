"""
    marginaldists(unique_nonempty_bins::Array{Int, 2},
                  invmeasure::Vector{Float64})

Compute margin distributions for a triangulation with an associated invariant
measure (`invmeasure`).

Computations are performed as follows:

- Find indices of non-empty bins beforehand (tuples, where each component gives
the index along each axis of the state space). This is information is calculated
beforehand, and is provided with the `nonempty_bins::Array{Int, 2}` argument.
- Compute the marginal distributions using the invariant measures associated
with the nonempty bins.

"""
function marginaldists(unique_nonempty_bins::Array{Int, 2},
                       invmeasure::Array{Float64, 1})
    dim = size(unique_nonempty_bins, 2)

    X2s = unique_nonempty_bins[:, 2] # Vector{Float64}
    X2s = reshape(X2s, length(X2s), 1) # Reshape to Array{Float64, 2}
    X1X2s = unique_nonempty_bins[:, 1:2]
    X2X3s = unique_nonempty_bins[:, 2:3]
    unique_X2s = unique(X2s, 1)
    unique_X1X2s = unique(X1X2s, 1)
    unique_X2X3s = unique(X2X3s, 1)
    JX2 = indexin_rows(X2s, unique_X2s)
    JX1X2 = indexin_rows(X1X2s, unique_X1X2s)
    JX2X3 = indexin_rows(X2X3s, unique_X2X3s)

    # X2
    PX2 = zeros(Float64, size(unique_X2s, 1))
    for i = 1:size(unique_X2s, 1)
        inds = find(JX2 .== i)
        PX2[i] = sum(invmeasure[inds])
    end

    Px2 = zeros(Float64, size(X2s, 1))
    for i = 1:size(X2s, 1)
        Px2[i] = PX2[JX2[i]]
    end

    # X1 and X2
    PX1X2 = zeros(Float64, size(unique_X1X2s, 1))
    for i = 1:size(unique_X1X2s, 1)
        inds = find(JX1X2 .== i)
        PX1X2[i] = sum(invmeasure[inds])
    end

    Px1x2 = zeros(Float64, size(X1X2s, 1))
    for i = 1:size(X1X2s, 1)
        Px1x2[i] = PX1X2[JX1X2[i]]
    end

    # X2 and X3
    PX2X3 = zeros(Float64, size(unique_X2X3s, 1))
    for i = 1:size(unique_X2X3s, 1)
        inds = find(JX2X3 .== i)
        PX2X3[i] = sum(invmeasure[inds])
    end

    Px2x3 = zeros(Float64, size(X2X3s, 1))
    for i = 1:size(X2X3s, 1)
        Px2x3[i] = PX2X3[JX2X3[i]]
    end

    return Px2, Px1x2, Px2x3
end
