"""
Draw point representatives from the simplices of a triangulation `t`.

Precedure:
1) Generate one point per simplex.
2) Points are generated from the interior or on the boundary of each simplex.
3) Points are drawn according to a uniform distribution.
"""
function point_representatives(t::T where T<:StateSpaceReconstruction.Partitioning.Triangulation)
    dim = size(t.points, 2)
    n_simplices = size(t.simplex_inds, 1)

    # Pre-allocate array to hold the points
    point_representatives = zeros(Float64, n_simplices, dim)

    # Loop over the rows of the simplex_inds array to access all the simplices.
    for i = 1:n_simplices
        simplex = t.points[t.simplex_inds[i, :], :]
        point_representatives[i, :] = Simplices.childpoint(simplex)
    end

    return point_representatives
end
