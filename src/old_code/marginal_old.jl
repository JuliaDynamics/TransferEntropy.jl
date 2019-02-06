using GroupSlices

"""
    marginaldists(nonempty_bins::Array{Int, 2},
                  invmeasure::Vector{Float64})

Compute marginal distributions for a triangulation with an associated invariant
measure (`invmeasure`).

Computations are performed as follows:

- Find indices of non-empty bins beforehand (tuples, where each component gives
the index along each axis of the state space). This is information is calculated
beforehand, and is provided with the `nonempty_bins::Array{Int, 2}` argument.
- Compute the marginal distributions using the invariant measures associated
with the nonempty bins.

"""
function marginaldists(nonempty_bins::Array{Int, 2},
                       invdist::Array{Float64, 1})

    #unique_nonempty_bins = unique(nonempty_bins, 1)

    dim = size(nonempty_bins, 2)


    X2s = nonempty_bins[:, 2] # Vector{Float64}
    X2s = reshape(X2s, length(X2s), 1) # Reshape to Array{Float64, 2}
    unique_X2s = unique(X2s, 1)
    J_X2 = indexin_rows(X2s, unique_X2s)


    marginal_x2 = zeros(Float64, size(unique_X2s, 1))

    for i = 1:size(unique_X2s, 1)
        multiplicity_i = find(J_X2 .== i)
        marginal_x2[i] = sum(invdist[multiplicity_i])
    end


    # X1 and X2
    X1X2s = nonempty_bins[:, 1:2]
    unique_X1X2s = unique(X1X2s, 1)
    J_X1X2 = indexin_rows(X1X2s, unique_X1X2s)

    marginal_x1x2 = zeros(Float64, size(unique_X1X2s, 1))
    for i = 1:size(unique_X1X2s, 1)
        multiplicity_i = J_X1X2 .== i
        marginal_x1x2[i] = sum(invdist[multiplicity_i])
    end

    # X2 and X3
    X2X3s = nonempty_bins[:, 2:3]
    unique_X2X3s = unique(X2X3s, 1)
    J_X2X3 = indexin_rows(X2X3s, unique_X2X3s)

    marginal_x2x3 = zeros(Float64, size(unique_X2X3s, 1))
    for i = 1:size(unique_X2X3s, 1)
        multiplicity_i = J_X2X3 .== i
        marginal_x2x3[i] = sum(invdist[multiplicity_i])
    end

    #@assert sum(marginal_x2) ≈ 1
    #@assert sum(marginal_x1x2) ≈ 1
    #@assert sum(marginal_x2x3) ≈ 1

    return marginal_x2, marginal_x1x2, marginal_x2x3,
            J_X2, J_X1X2, J_X2X3

end
