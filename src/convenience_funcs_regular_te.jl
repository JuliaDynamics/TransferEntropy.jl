

"""
    transferentropy(source::AbstractArray{Real, 1}, response::AbstractArray{Real, 1}; 
        dim = 3, η = 1, τ = 1, 
        estimator = VisitationFrequency(), 
        n_subdivs = 3)

Compute transfer entropy from `source` to `target` using a `dim`-dimensional delay 
reconstruction with prediction lag `η` and embedding lag `τ`,  inferring appropriate 
bin sizes from the time series length. Extra dimensions are assigned to the ``T_{pp}`` 
component of the delay reconstruction (see [`TEVars`](@ref)), while 1D embeddings 
are used for the ``T_f`` and ``S_pp`` component. 

Returns one transfer entropy estimate per partition.
"""
function transferentropy(source::AbstractArray{Real, 1}, response::AbstractArray{Real, 1}; 
    dim = 3, η = 1, τ = 1, 
    estimator = VisitationFrequency(), 
    n_subdivs = 3)

    dim >= 3 || throw(ArgumentError("dim must be 3 or higher for regular TE"))

    # Distribute dimensions with as (Tf -> k = 1), (Tpp -> l = dim - 2), (Spp -> m = 1),
    # so that it is the states present/past of the target variable that gets a longer 
    # history.
    k, l, m = 1, dim - 2, 1
    pts, vars = te_embed(source, response, k, l, m, η = η, τ = τ)

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
    transferentropy(source::AbstractArray{Real, 1}, response::AbstractArray{Real, 1}, 
        binning_scheme::Union{RectangularBinning, AbstractArray{RectangularBinning}}; 
        dim = 3, η = 1, τ = 1, estimator = VisitationFrequency())

Compute transfer entropy from `source` to `target` using a `dim`-dimensional delay 
reconstruction with prediction lag `η` and embedding lag `τ`, inferring appropriate bin 
sizes from time series length. Extra dimensions are assigned to the ``T_{pp}`` component of 
the delay reconstruction (see [`TEVars`](@ref)), while 1D embeddings 
are used for the ``T_f`` and ``S_pp`` component. 

Returns one transfer entropy estimate per partition.
"""
function transferentropy(source::AbstractArray{Real, 1}, response::AbstractArray{Real, 1}, 
    binning_scheme::Union{RectangularBinning, AbstractArray{RectangularBinning}}; 
    dim = 3, η = 1, τ = 1, estimator = VisitationFrequency())

    dim >= 3 || throw(ArgumentError("dim must be 3 or higher for regular TE"))

    # Distribute dimensions with as (Tf -> k = 1), (Tpp -> l = dim - 2), (Spp -> m = 1),
    # so that it is the states present/past of the target variable that gets a longer 
    # history.
    k, l, m = 1, dim - 2, 1
    pts, vars = te_embed(source, response, k, l, m, η = η, τ = τ)

    # Compute TE over different partitions
    # ====================================
    if binning_scheme isa RectangularBinning
        return transferentropy(pts, vars, binning_scheme, estimator)
    else
        return map(b -> transferentropy(pts, vars, b, estimator), binning_scheme)
    end
end


"""
    transferentropy(source::AbstractArray{Real, 1}, response::AbstractArray{Real, 1}, 
        k::Int, l::Int, m::Int; 
        η = 1, τ = 1, estimator = VisitationFrequency(), n_subdivs = 3)

Compute transfer entropy from `source` to `target` using a `k + l + m`-dimensional delay 
reconstruction with `k` being the dimension of the ``T_f`` component, `l` the 
dimension of the ``T_{pp}`` component and `m` the dimension of the ``S_{pp}`` component
(see [`TEVars`](@ref)), using prediction lag `η` and embedding lag `τ`, 
inferring appropriate bin sizes from time series length. 

Returns one transfer entropy estimate per partition.
"""
function transferentropy(source::AbstractArray{Real, 1}, response::AbstractArray{Real, 1}, 
        k::Int, l::Int, m::Int; 
        η = 1, τ = 1, estimator = VisitationFrequency(), n_subdivs = 3)

    k + l + m >= 3 || throw(ArgumentError("`dim = k + l + m` must be 3 or higher for regular TE"))

    pts, vars = te_embed(source, response, k, l, m, η = η, τ = τ)

    # Determine appropriate binnings from time series length (according to 
    # Krakovska et al. (2018)'s recommendations)
    L = length(source)
    n_subdivisions_coarsest = ceil(Int, L^(1/(k + l + m + 1)))
    n_subdivisions_finest = n_subdivisions_coarsest + n_subdivs
    binning_scheme = map(n-> RectangularBinning(n), n_subdivisions_coarsest:n_subdivisions_finest)

    # Compute TE over different partitions
    # ====================================
    tes = map(b -> transferentropy(pts, vars, b, estimator), binning_scheme)

    return tes
end




"""
    transferentropy(source::AbstractArray{Real, 1}, response::AbstractArray{Real, 1}, 
        binning_scheme::Union{RectangularBinning, AbstractArray{RectangularBinning}}, 
        k::Int, l::Int, m::Int; 
        dim = 3, η = 1, τ = 1, estimator = VisitationFrequency())

Compute transfer entropy from `source` to `target` using a `k + l + m`-dimensional delay 
reconstruction with `k` being the dimension of the ``T_f`` component, `l` the dimension 
of the ``T_{pp}`` component and `m` the dimension of the ``S_{pp}`` component
(see [`TEVars`](@ref)), using prediction lag `η` and embedding lag `τ`, 
explicitly specifying the partition(s). 

Returns one transfer entropy estimate per partition.
"""
function transferentropy(source::AbstractArray{Real, 1}, response::AbstractArray{Real, 1}, 
        binning_scheme::Union{RectangularBinning, AbstractArray{RectangularBinning}}, 
        k::Int, l::Int, m::Int; 
        dim = 3, η = 1, τ = 1, estimator = VisitationFrequency())

    k + l + m >= 3 || throw(ArgumentError("`dim = k + l + m` must be 3 or higher for regular TE"))

    # Distribute dimensions with as (Tf -> k = 1), (Tpp -> l = dim - 2), (Spp -> m = 1),
    # so that it is the states present/past of the target variable that gets a longer 
    # history.
    pts, vars = te_embed(source, response, k, l, m, η = η, τ = τ)

    # Compute TE over different partitions
    # ====================================
    if binning_scheme isa RectangularBinning
        return transferentropy(pts, vars, binning_scheme, estimator)
    else
        return map(b -> transferentropy(pts, vars, b, estimator), binning_scheme)
    end
end

export transferentropy