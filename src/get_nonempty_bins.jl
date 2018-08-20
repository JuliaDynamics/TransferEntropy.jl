"""
Superimpose a grid on a triangulation with an associated invariant measure, and
find the indices of boxes in the grid with nonzero measure.

The grid consists of rectangular boxes with constant size along each dimension.
The number of bins along each dimension is given by `δ::Vector{Int}`, where
each element given the number of bins along each dimension (in the order
specified by the triangulation).

## Arguments
`pts::Array{Float, 2}` is an array one points, each contained within a
unique simplex of the triangulation (size n_simplices x dim array).

`invdist::Array{Float64, 1}` is the invariant distribution on the
triangulation

`δ::Vector{Float64}` are the number of bins along each dimension. The algorithm
divides each axis into equidistant bins from min(pts[:, i]) to
max(pts[:, i]).

"""
function get_nonempty_bins(pts::Array{Float64, 2},
                        invdist::Array{Float64, 1},
                        δ::Vector{Int})

    # If a centroid has zero measure associated with it, don't consider it.
    # Do this by finding indices of pts (row indices in the pts
    # array) of simplices with nonzero measure.
    # positivemeasure_inds = find(invdist .> 0)
    # pts = pts[positivemeasure_inds, :]
    # invdist = invdist[positivemeasure_inds]

    n_simplices = size(pts, 1)
    dim = size(pts, 2)

    # The number of grids specified must match the number of dimensions.
    #@assert dim == length(δ)

    # Initialise matrix holding the non-zero entries of the joint distribution.
    # Each row contains the indices
    nonempty_bins = zeros(Int, n_simplices, dim)


    # For each simplex, find the index of the bin its point lies in for each
    # of the dimensions. For this, we need to know where to locate the bins.
    startvals = [minimum(pts[:, i]) for i in 1:dim]
    endvals = [maximum(pts[:, i]) for i in 1:dim]
    ranges = [(endvals[i] - startvals[i]) for i in 1:dim]

    # For each simplex
    for i = 1:n_simplices
        # that has measure invdist[i]
        for j = 1:dim
            # Find the bin index along dimension j its point representative
            # falls in
            stepsize = ranges[j] / δ[j]
            pos_along_range = pts[i, j] - startvals[j]

            if pos_along_range == 0
                nonempty_bins[i, j] = 1
            else
                nonempty_bins[i, j] = ceil(Int, pos_along_range / stepsize)
            end
        end
    end
    return nonempty_bins
end

"""
    `get_nonempty_bins_abs(pts::Array{Float64, 2},
                    invdist::Array{Float64, 1},
                    δ::Vector{Int})`

Get nonempty bins given a vector absolute bin sizes `δ` along each dimension.

This function is useful if you want to compute TE only down to a given bin size, and
differs from `get_nonempty_bins`, which uses a regularly spaced grid along each axis.
"""
function get_nonempty_bins_abs(pts::Array{Float64, 2},
                    invdist::Array{Float64, 1},
                    δ::Vector{Float64})

    # If a centroid has zero measure associated with it, don't consider it.
    # Do this by finding indices of pts (row indices in the pts
    # array) of simplices with nonzero measure.
    positivemeasure_inds = find(invdist .> 0)
    pts = pts[positivemeasure_inds, :]
    invdist = invdist[positivemeasure_inds]

    n_simplices = size(pts, 1)
    dim = size(pts, 2)

    # The number of grids specified must match the number of dimensions.
    #@assert dim == length(δ)

    # Initialise matrix holding the non-zero entries of the joint distribution.
    # Each row contains the indices
    nonempty_bins = zeros(Int, n_simplices, dim)

    # Initialise column vector holding the invariant invmeasure of the boxes
    # corresponding to rows in 'joint'.
    invmeasure = zeros(Float64, n_simplices)

    # For each simplex, find the index of the bin its centroid lies in for each
    # of the dimensions. For this, we need to know where to locate the bins.
    startvals = [minimum(pts[:, i]) for i in 1:dim]
    endvals = [maximum(pts[:, i]) for i in 1:dim]
    ranges = [(endvals[i] - startvals[i]) for i in 1:dim]

    for i = 1:n_simplices
        invmeasure[i] = invdist[i]
        for j = 1:dim
            pos_along_range = pts[i, j] - startvals[j]
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
