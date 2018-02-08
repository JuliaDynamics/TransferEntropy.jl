"""
Superimpose a grid on a triangulation with an associated invariant measure, and
find the indices of boxes in the grid with nonzero measure.

The grid consists of rectangular boxes with constant size along each dimension.
The number of bins along each dimension is given by `δ::Vector{Int}`, where
each element given the number of bins along each dimension (in the order
specified by the triangulation).

## Arguments
`centroids::Array{Float, 2}` are centroids of the simplices forming the
triangulation; this is an with size n_simplices x dim.

`invariantdist::Array{Float64, 1}` is the invariant distribution on the
triangulation

`δ::Vector{Float64}` are the number of bins along each dimension. The algorithm
divides each axis into equidistant bins from min(centroids[:, i]) to
max(centroids[:, i]).

"""
function get_nonempty_bins(centroids::Array{Float64, 2},
                    invariantdist::Array{Float64, 1},
                    δ::Vector{Int})

    # If a centroid has zero measure associated with it, don't consider it.
    # Do this by finding indices of centroids (row indices in the centroids
    # array) of simplices with nonzero measure.
    nonzeromeasure_inds = find(invariantdist .> 0)
    centroids = centroids[nonzeromeasure_inds, :]
    invariantdist = invariantdist[nonzeromeasure_inds]

    n_simplices = size(centroids, 1)
    dim = size(centroids, 2)

    # The number of grids specified must match the number of dimensions.
    @assert dim == length(δ)

    # Initialise matrix holding the non-zero entries of the joint distribution.
    # Each row contains the indices
    nonempty_bins = zeros(Int, n_simplices, dim)

    # Initialise column vector holding the invariant invmeasure of the boxes
    # corresponding to rows in 'joint'.
    invmeasure = zeros(Float64, n_simplices)

    # For each simplex, find the index of the bin its centroid lies in for each
    # of the dimensions. For this, we need to know where to locate the bins.
    startvals = [minimum(centroids[:, i]) for i in 1:dim]
    endvals = [maximum(centroids[:, i]) for i in 1:dim]
    ranges = [(endvals[i] - startvals[i]) for i in 1:dim]

    for i = 1:n_simplices
        invmeasure[i] = invariantdist[i]
        for j = 1:dim
            stepsize = ranges[j] / δ[j]
            pos_along_range = centroids[i, j] - startvals[j]

            if pos_along_range == 0
                nonempty_bins[i, j] = 1
            else
                nonempty_bins[i, j] = ceil(Int, pos_along_range / stepsize)
            end
        end
    end
    return nonempty_bins, invmeasure
end

"""
    `get_nonempty_bins_abs(centroids::Array{Float64, 2},
                    invariantdist::Array{Float64, 1},
                    δ::Vector{Int})`

Get nonempty bins given a vector absolute bin sizes `δ` along each dimension.

This function is useful if you want to compute TE only down to a given bin size, and
differs from `get_nonempty_bins`, which uses a regularly spaced grid along each axis.
"""
function get_nonempty_bins_abs(centroids::Array{Float64, 2},
                    invariantdist::Array{Float64, 1},
                    δ::Vector{Float64})

    # If a centroid has zero measure associated with it, don't consider it.
    # Do this by finding indices of centroids (row indices in the centroids
    # array) of simplices with nonzero measure.
    nonzeromeasure_inds = find(invariantdist .> 0)
    centroids = centroids[nonzeromeasure_inds, :]
    invariantdist = invariantdist[nonzeromeasure_inds]

    n_simplices = size(centroids, 1)
    dim = size(centroids, 2)

    # The number of grids specified must match the number of dimensions.
    @assert dim == length(δ)

    # Initialise matrix holding the non-zero entries of the joint distribution.
    # Each row contains the indices
    nonempty_bins = zeros(Int, n_simplices, dim)

    # Initialise column vector holding the invariant invmeasure of the boxes
    # corresponding to rows in 'joint'.
    invmeasure = zeros(Float64, n_simplices)

    # For each simplex, find the index of the bin its centroid lies in for each
    # of the dimensions. For this, we need to know where to locate the bins.
    startvals = [minimum(centroids[:, i]) for i in 1:dim]
    endvals = [maximum(centroids[:, i]) for i in 1:dim]
    ranges = [(endvals[i] - startvals[i]) for i in 1:dim]

    for i = 1:n_simplices
        invmeasure[i] = invariantdist[i]
        for j = 1:dim
            pos_along_range = centroids[i, j] - startvals[j]
            if pos_along_range == 0
                nonempty_bins[i, j] = 1
            else
                nonempty_bins[i, j] = ceil(Int, pos_along_range / δ[j])
            end
        end
    end

    # Return the indices of the nonempty bins and the corresponding invariant measure
    return nonempty_bins, invmeasure, ranges
end
