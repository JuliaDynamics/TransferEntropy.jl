import LinearAlgebra: transpose!
import ..Simplices: subsample_coeffs
import StaticArrays: SVector, MVector, SMatrix, MMatrix

"""
    interiorpts(simplex_vertices, n_fillpoints = 100, sample_randomly = true)

Generate a total of `n_fillpoints` points lying inside the simplex defined 
by `simplex_vertices` (a vector of column vectors).

If `sample_randomly = true`, then points are generated randomly inside each
subsimplex. If `sample_randomly = false` (default), then points are generated
as the centroids of the subsimplices resulting from a shape-preserving
refinement of the simplices in the convex hull.

## Arguments
- **``sᵢ``**: An array where each column is a simplex vertex.
- **``n_fillpoints``**: The number of points to fill the simplex with. 
- **``sample_randomly``**: Should points be sampled randomly within the
    intersection? Default is `sample_randomly = false`, which inserts points
    inside the intersecting volume in a regular manner.
"""
function interiorpts(simplex_vertices, n_fillpoints::Int = 100, sample_randomly::Bool = true)
    dim = length(simplex_vertices[1])
    T = typeof(simplex_vertices[1][1])
    s = SMatrix{dim+1, dim}(transpose(hcat(simplex_vertices...,)))
    
    #s = transpose(hcat(simplex_vertices...,))
    # Generate a set of a minimum of `n_fillpoints` sets of convex coefficients, so that 
    # interior points of the simplex can be constructed as a linear combination 
    # of `simplex_vertices` using any set of those coefficients.
    convex_coeffs = subsample_coeffs(dim, n_fillpoints, sample_randomly)
    
    fillpts = Vector{SVector{dim, T}}(undef, n_fillpoints)
    C = zeros(1, dim+1)
    @inbounds for i = 1:n_fillpoints
        for k = 1:dim+1
            C[k] = convex_coeffs[k, i]
        end
        
        fillpts[i] = SMatrix{1, dim+1}(C) * s
    end
    
    return fillpts
end