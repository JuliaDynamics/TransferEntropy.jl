"""
    te_from_triangulation(centroids::Array{Int, 2}, invmeasure::Vector{Float64})

Compute transfer entropy from pre-provided joint and marginal distributions.
`n` is the number of equally sized rectangular bins to use.
"""
function te_from_triangulation(centroids::Array{Float64, 2},
                               invariantdistribution::Array{Float64, 1},
                               n::Int)

    # Find non empty bins and their measure
    nonempty_bins, measure = get_nonempty_bins(centroids,
                                                invariantdistribution,
                                                [n, n, n])

    # Compute joint distribution.
    joint = jointdist(nonempty_bins, measure)

    # Compute marginal distributions.
    Py, Pxy, Pyz = marginaldists(unique(nonempty_bins, 1), measure)

    # Initialise transfer entropy to 0. Because the measure of the bins
    # are guaranteed to be nonnegative, transfer entropy is also guaranteed
    # to be nonnegative.
    te = 0.0

    # Compute transfer entropy.
    for i = 1:length(joint)
        te += joint[i] * log(joint[i] * Py[i] / (Pyz[i] * Pxy[i]))
    end

    return te
end
