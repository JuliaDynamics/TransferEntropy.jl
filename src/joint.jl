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
function jointdist(nonempty_bins::Array{Int, 2}, invmeasure::Array{Float64, 1})

    unique_bins = unique(nonempty_bins, 1)
    Pjoint = Array(Float64, size(unique_bins, 1))
    Jjoint = indexin_rows(nonempty_bins, unique_bins)

    for i = 1:size(unique_bins, 1)
        filljoint!(Pjoint, Jjoint, invmeasure, i)
    end

    for i = 1:size(unique_bins, 1)
        inds = find(Jjoint .== i)
        Pjoint[i] .= sum(invmeasure[inds])
    end

    return Pjoint
end

function filljoint!(Pjoint, Jjoint, invmeasure, i)
    inds = find(Jjoint .== i)
    Pjoint[i] .= sum(invmeasure[inds])
end


"""
Given indices of non-empty bins and their corresponding measure
"""
function jdist(bininds::Array{Int, 2}, measure::Array{Float64, 1})


end
