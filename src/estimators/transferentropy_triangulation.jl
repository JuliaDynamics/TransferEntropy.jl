"""
	point_representatives(t::AbstractTriangulation)

Draw point representatives from the simplices of a triangulation `t`.

Precedure:
1) Generate one point per simplex.
2) Points are generated from the interior or on the boundary of each simplex.
3) Points are drawn according to a uniform distribution.
"""
function point_representatives(t::AbstractTriangulation)
    dim = size(t.points, 2)
    n_simplices = size(t.simplex_inds, 1)

    # Pre-allocate array to hold the points
    point_representatives = zeros(Float64, n_simplices, dim)

    # Loop over the rows of the simplex_inds array to access all the simplices.
    for i = 1:n_simplices
        simplex = t.points[t.simplex_inds[i, :], :]
        point_representatives[i, :] = childpoint(simplex)
    end

    return point_representatives
end

"""
	point_representatives(t::AbstractTriangulation)

Draw multiple point representatives for each of the simplices in the
triangulation `t`. Each returned point is a column vector.

Precedure:
1) Generate `n` points per simplex.
2) Points are generated from the interior or on the boundary of each simplex.
3) Points are drawn according to a uniform distribution.
"""
function point_representatives(t::AbstractTriangulation, n::Int)
    dim = size(t.points, 2)
    n_simplices = size(t.simplex_inds, 1)

    # Pre-allocate array to hold the points
    insidepts = zeros(Float64, dim, n, n_simplices)

    point_representatives = Vector{Array{Float64, 2}}(n)

    # Loop over the rows of the simplex_inds array to access all the simplices.
    for i = 1:n_simplices
        simplex = t.points[t.simplex_inds[i, :], :]
        for j=1:n
            point_representatives[:, j, i] = childpoint(simplex)
        end
    end

    return point_representatives
end

"""
Superimpose a grid on a triangulation with an associated invariant measure, and
find the (approximate) indices of boxes in the grid with nonzero measure.

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

function indexin_rows(A1::Array{T, 2}, A2::Array{T, 2}) where {T<:Number}
    inds = []
    for j = 1:size(A1, 1)
        for i = 1:size(A2, 1)
            if all(A1[j, :] .== A2[i, :])
                push!(inds, i)
            end
        end
    end
    return inds
end



"""
    transferentropy_triang(
        t::AbstractTriangulation,
        invdist::PerronFrobenius.InvariantDistribution,
        n_bins::Int,
        n_reps::Int
    ) -> Vector{Float64}

Compute transfer entropy from a triangulation `t` with an associated invariant
distribution over the simplices. This distribution
gives the probability that trajectories - in the long term - will visit regions
of the state space occupied by each simplex.

A single transfer entropy estimate is obtained by sampling one point within each
simplex. We then overlay a rectangular grid, being regular along each
dimension, with `n_bins` determining the number of equally sized chunks to divide
the grid into along each dimension.

The probabily of a grid cell as the weighted sum of the generated points falling
inside that bin, where the weights are the invariant probabilities associated
with the simplices to which the points belong.

Joint and marginal probability distributions are then obtained by keeping
relevant axes fixed, summing over the remaining axes.

Repeating this procedure `n_reps` times, we obtain a distribution of
TE estimates for this bin size.
"""
function transferentropy_transferoperator_triang(
        t::AbstractTriangulation,
        invdist::PerronFrobenius.InvariantDistribution,
        n_bins::Int,
        n_reps::Int
    )

    # Initialise transfer entropy estimates to 0. Because the measure of the
    # bins are guaranteed to be nonnegative, transfer entropy is also guaranteed
    # to be nonnegative.
    TE_estimates = zeros(Float64, n_reps)

    for i = 1:n_reps
        # Represent each simplex as a single point. We can do this because
        # within the region of the state space occupied by each simplex, points
        # are indistinguishable from the point of view of the invariant measure.
        # However, when we superimpose the grid, the position of the points
        # we choose will influence the resulting marginal distributions.
        # Therefore, we have to repeat this procedure several times to get an
        # accurate transfer entropy estimate.

        positive_measure_inds = find(invdist.dist .> 1/10^8)

        # Find non-empty bins and compute their measure.
        nonempty_bins = get_nonempty_bins(
            #t.centroids[positive_measure_inds, :],
    		point_representatives(t)[positive_measure_inds, :],
    		invdist.dist[positive_measure_inds],
    		[n_bins, n_bins, n_bins]
        )


        # Compute the joint and marginal distributions.
        Pjoint = jointdist(nonempty_bins, invdist.dist[positive_measure_inds])
        Py, Pxy, Pyz, Jy, Jxy, Jyz = marginaldists(nonempty_bins, invdist.dist[positive_measure_inds])

        # Use base 2 for the logarithm, so that we get transfer entropy in bits
        for k = 1:size(Pjoint, 1)
            TE_estimates[i] += Pjoint[k] *
                log(2, (Pjoint[k] * Py[Jy[k]]) / (Pxy[Jxy[k]] * Pyz[Jyz[k]]) )
        end
    end

    return TE_estimates
end

tetotri = transferentropy_transferoperator_triang
