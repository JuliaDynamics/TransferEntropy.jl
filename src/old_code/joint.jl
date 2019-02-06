"""
    jointdist(nonempty_bins::Array{Int, 2}, invmeasure::Vector{Float64})

Compute joint distribution for a triangulation with an associated invariant
measure (`invmeasure`).

Computations are performed as follows:

- Find indices of non-empty bins beforehand (tuples, where each component gives
the index along each axis of the state space). This is information is calculated
beforehand, and is provided with the `nonempty_bins::Array{Int, 2}` argument.
- Compute the joint distribution using the invariant measures associated with
the nonempty bins.

"""
function jointdist(nonempty_bins::Array{Int, 2}, invdist::Array{Float64, 1})

    unique_nonempty_bins = unique(nonempty_bins, 1)
    J = indexin_rows(nonempty_bins, unique_nonempty_bins)


    joint = zeros(Float64, size(unique_nonempty_bins, 1))

        for i = 1:size(unique_nonempty_bins, 1)
            multiplicity_i = find(J .== i)
            joint[i] = sum(invdist[multiplicity_i])
        end

    return joint
end
