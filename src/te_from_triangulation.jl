"""
Compute transfer entropy from pre-provided joint and marginal distributions.
"""
function te_from_triangulation(centroids::Array{Int, 2},
                               invmeasure::invmeasure::Vector{Float64})

    # Find non empty bins.
    nonempty_bins, invmeasure = get_nonempty_bins(centroids,
                                                invariantdistribution,
                                                [n, n, n])

    # Compute joint distribution.
    joint = jointdist(nonempty_bins, measure)

    # Compute marginal distributions.
    Py, Pxy, Pxz = marginaldists(unique(nonempty_bins, 1), measure)

    # Initialise transfer entropy to 0. Because the measure of the bins
    # are guaranteed to be nonnegative, transfer entropy is also guaranteed
    # to be nonnegative.
    TE = 0

    # Compute transfer entropy.
    for i = 1:size(unique(nonempty_bins, 1), 1)
        Pxyz = joint[i]
        Py = Py[i]
        Pxy = Pxy[i]
        Pyz = Pxz[i]
        TE = TE + Pxyz * log(Pxyz * Py / (Pyz * Pxy))
    end

    return TE
end
