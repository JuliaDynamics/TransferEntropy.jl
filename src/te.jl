"""
Compute transfer entropy from pre-provided joint and marginal distributions.
"""
function te_from_joint_and_marginals(
    unique_nonempty_bins::Array{Int, 2},
    joint::Array{Float64, 1},
    marginals::Tuple{Array{Float64, 1}, Array{Float64, 1}, Array{Float64, 1}})

    TE = 0

    for i = 1:size(unique_nonempty_bins, 1)
        Pxyz = joint[i]
        Py = marginals[1][i]
        Pxy = marginals[2][i]
        Pyz = marginals[3][i]

        TE = TE + Pxyz * log(Pxyz * Py / (Pyz * Pxy))
    end

    return TE
end
