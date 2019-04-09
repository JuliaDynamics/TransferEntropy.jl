
"""
    transferentropy(source::AbstractArray{<:Real, 1}, 
        response::AbstractArray{<:Real, 1}, 
        cond::AbstractArray{<:Real, 1},
        binning_scheme::Union{RectangularBinning, AbstractArray{RectangularBinning}};
        dim = 4, η = 1, τ = 1, estimator = VisitationFrequency())

Compute transfer entropy from `source` to `target` conditioned on `cond` 
using a `dim`-dimensional delay 
reconstruction with prediction lag `η` and embedding lag `τ`, inferring appropriate bin 
sizes from time series length. Extra dimensions are assigned to the ``T_{pp}`` component of 
the delay reconstruction (see [`TEVars`](@ref)), while 1D embeddings 
are used for the ``T_f``, ``S_{pp}`` and ``C_{pp}`` components. 
    
Returns one transfer entropy estimate per partition.
"""
function transferentropy(source::AbstractArray{<:Real, 1}, 
    response::AbstractArray{<:Real, 1}, 
    cond::AbstractArray{<:Real, 1},
    binning_scheme::Union{RectangularBinning, AbstractArray{RectangularBinning}};
    dim = 4, η = 1, τ = 1, estimator = VisitationFrequency())

    dim >= 4 || throw(ArgumentError("`dim` must be 4 or higher for conditional TE"))

    # Distribute dimensions with as (Tf -> k = 1), (Tpp -> l = dim - 2), (Spp -> m = 1), (Cpp -> n = 1)
    # so that it is the states present/past of the target variable that gets a longer 
    # history.
    k, l, m, n = 1, dim - 2, 1, 1
    pts, vars = te_embed(source, response, cond, k, l, m, n, η = η, τ = τ)

    # Compute TE over different partitions
    # ====================================
    if binning_scheme isa RectangularBinning
        return transferentropy(pts, vars, binning_scheme, estimator)
    else
        return map(b -> transferentropy(pts, vars, b, estimator), binning_scheme)
    end
end


"""
    transferentropy(source::AbstractArray{<:Real, 1}, 
        response::AbstractArray{<:Real, 1}, 
        cond::AbstractArray{<:Real, 1};
        dim = 4, η = 1, τ = 1, estimator = VisitationFrequency(), n_subdivs = 3)

Compute transfer entropy from `source` to `target` conditioned on `cond`,
using a `dim`-dimensional delay 
reconstruction, using 
prediction lag `η` and embedding lag `τ`, 
inferring appropriate bin sizes from time series length. 
Extra dimensions are assigned to the ``T_{pp}`` component of 
the delay reconstruction (see [`TEVars`](@ref)), while 1D embeddings 
are used for the ``T_f``, ``S_{pp}`` and ``C_{pp}`` components. 

Returns one transfer entropy estimate per partition.
"""
function transferentropy(source::AbstractArray{<:Real, 1}, 
        response::AbstractArray{<:Real, 1}, 
        cond::AbstractArray{<:Real, 1};
        dim = 4, η = 1, τ = 1, estimator = VisitationFrequency(), n_subdivs = 3)

    dim >= 4 || throw(ArgumentError("`dim` must be 4 or higher for conditional TE"))

    # Distribute dimensions with as (Tf -> k = 1), (Tpp -> l = dim - 2), (Spp -> m = 1), (Cpp -> n = 1)
    # so that it is the states present/past of the target variable that gets a longer 
    # history.
    k, l, m, n = 1, dim - 2, 1, 1
    pts, vars = te_embed(source, response, cond, k, l, m, n, η = η, τ = τ)

    # Determine appropriate binnings from time series length (according to 
    # Krakovska et al. (2018)'s recommendations)
    L = length(source)
    n_subdivisions_coarsest = ceil(Int, L^(1/(dim + 1)))
    n_subdivisions_finest = n_subdivisions_coarsest + n_subdivs
    binning_scheme = map(n-> RectangularBinning(n), n_subdivisions_coarsest:n_subdivisions_finest)

    # Compute TE over different partitions
    # ====================================
    tes = map(b -> transferentropy(pts, vars, b, estimator), binning_scheme)

    return tes
end


"""
    transferentropy(source::AbstractArray{<:Real, 1}, 
        response::AbstractArray{<:Real, 1}, 
        cond::AbstractArray{<:Real, 1},
        binning_scheme::Union{RectangularBinning, AbstractArray{RectangularBinning}},
        k::Int, l::Int, m::Int, n::Int; η = 1, τ = 1, estimator = VisitationFrequency())

Compute transfer entropy from `source` to `target` conditioned on `cond`, 
using a `k + l + m + n`-dimensional delay 
reconstruction with with `k` being the dimension of the ``T_f`` component, `l` the dimension 
of the ``T_{pp}`` component, `m` the dimension of the ``S_{pp}`` component and 
`n` the dimension of the ``C_{pp}`` component (see [`TEVars`](@ref)), 
using prediction lag `η` and embedding lag `τ`, inferring appropriate bin 
sizes from time series length.

Returns one transfer entropy estimate per partition.
"""
function transferentropy(source::AbstractArray{<:Real, 1}, 
        response::AbstractArray{<:Real, 1}, 
        cond::AbstractArray{<:Real, 1},
        binning_scheme::Union{RectangularBinning, AbstractArray{RectangularBinning}},
        k::Int, l::Int, m::Int, n::Int; η = 1, τ = 1, estimator = VisitationFrequency())

    k + l + m + n >= 4 || throw(ArgumentError("`dim = k + l + m + n` must be 4 or higher for conditional TE"))

    # Distribute dimensions with as (Tf -> k = 1), (Tpp -> l = dim - 2), (Spp -> m = 1), (Cpp -> n = 1)
    # so that it is the states present/past of the target variable that gets a longer 
    # history.
    pts, vars = te_embed(source, response, cond, k, l, m, n, η = η, τ = τ)

    # Compute TE over different partitions
    # ====================================
    if binning_scheme isa RectangularBinning
        return transferentropy(pts, vars, binning_scheme, estimator)
    else
        return map(b -> transferentropy(pts, vars, b, estimator), binning_scheme)
    end
end

"""
    transferentropy(source::AbstractArray{<:Real, 1}, 
        response::AbstractArray{<:Real, 1},
        cond::AbstractArray{<:Real, 1},
        k::Int, l::Int, m::Int, n::Int;
        η = 1, τ = 1, estimator = VisitationFrequency(), n_subdivs = 3)

Compute transfer entropy from `source` to `target` conditioned on `cond` 
using a `k + l + m + n`-dimensional delay 
reconstruction with prediction lag `η` and embedding lag `τ`,  inferring appropriate 
bin sizes from the time series length (see [`TEVars`](@ref)).

Returns one transfer entropy estimate per partition.
"""
function transferentropy(source::AbstractArray{<:Real, 1}, 
        response::AbstractArray{<:Real, 1},
        cond::AbstractArray{<:Real, 1},
        k::Int, l::Int, m::Int, n::Int;
        η = 1, τ = 1, estimator = VisitationFrequency(), n_subdivs = 3)

    k + l + m + n >= 4 || throw(ArgumentError("`dim = k + l + m + n` must be 4 or higher for conditional TE"))

    # Distribute dimensions with as (Tf -> k = 1), (Tpp -> l = dim - 2), (Spp -> m = 1), (Cpp -> n = 1)
    # so that it is the states present/past of the target variable that gets a longer 
    # history.
    pts, vars = te_embed(source, response, cond, k, l, m, n, η = η, τ = τ)

    # Determine appropriate binnings from time series length (according to 
    # Krakovska et al. (2018)'s recommendations)
    L = length(source)
    n_subdivisions_coarsest = ceil(Int, L^(1/(k + l + m + n + 1)))
    n_subdivisions_finest = n_subdivisions_coarsest + n_subdivs
    binning_scheme = map(n-> RectangularBinning(n), n_subdivisions_coarsest:n_subdivisions_finest)

    # Compute TE over different partitions
    # ====================================
    tes = map(b -> transferentropy(pts, vars, b, estimator), binning_scheme)

    return tes
end

export transferentropy