using CausalityToolsBase
using DelayEmbeddings

"""
    te_embed(source::AbstractArray{T, 1}, target::AbstractArray{T, 1},
        k = 1, l = 1, m = 1, n = 0, η = 1, τ = 1)

Create an appropriate custom delay reconstruction from `source` and `target` for the
purpose of computing transfer entropy from `source` to `target`. 

# Delay reconstruction

Denote the source time series by `s(t)` and target time series by `t(t)`. Then the 
embedding vectors are on the the form

``(T_{f}, T_{pp}, S_{pp})`` where

``T_f = (t(t + k*η), ..., t(t + 2*η), t(t + η))``,
``T_pp = (t(t), t(t - τ), t(t - 2*τ), ..., t(t - lτ))``,
``S_pp = (s(t), s(t - τ), s(t - 2*τ), ..., s(t - mτ))`` 

and the total dimension of the reconstructed space is `k + l + m`.

# Arguments 

- **`source`**: A source data series. 
- **`target`**: A target data series. 
- **`k`**: The dimension of the ``T_f`` component of the delay reconstruction.
- **`l`**: The dimension of the ``T_{pp}`` component of the delay reconstruction.
- **`m`**: The dimension of the ``S_{pp}`` component of the delay reconstruction.

# Keyword arguments 
- **`η`**: The prediction lag that goes into the ``T_f`` component of the 
    delay reconstruction.
- **`τ`**: The embedding lag that goes into the ``T_{pp}`` and ``S_{pp}`` 
    components of the delay reconstruction.

# Examples 

```julia 
x, y, z = rand(100), rand(100), rand(100)

# Get embedding points and instructions for computing marginals.

# TE(x -> y)
# ==================
pts, vars = te_embed(x, y) # default k = l = m = 1
pts, vars = te_embed(x, y, 1, 1, 1) # specify k, l and m

pts, vars = te_embed(x, y, η = 2) # specify prediction lag
pts, vars = te_embed(x, y, τ = 2) # specify embedding lag

# Compute TE 
# ======================
transferentropy(pts, vars, RectangularBinning(5), VisitationFrequency())
```
"""
function te_embed(source::AbstractArray{T, 1}, target::AbstractArray{T, 1},
        k = 1, l = 1, m = 1; η = 1, τ = 1) where T

    length(source) == length(target) || throw(ArgumentError("length(source) != length(target)"))
    k >= 1 || throw(ArgumentError("k=$k must be >= 1 (so that Tf has entries)"))
    l >= 1 || throw(ArgumentError("l=$l must be >= 1 (so that Tpp has entries)"))
    m >= 1 || throw(ArgumentError("m=$m must be >= 1 (so that Spp has entries)"))
    k + l + m >= 3 || throw(ArgumentError("k + l + m must be at least 3 for meaningful TE computations"))
    
    # Positions (ks and ls both refer to the target time series, 
    # while ms refer to the source time series).
    pos_ks = repeat([2], k) 
    pos_ls = repeat([2], l)
    pos_ms = repeat([1], m)
    pos = Positions([pos_ks; pos_ls; pos_ms])
    
    # Lags
    lags_ks = [k*η for k in k:-1:1]
    lags_ls = [-l*τ for l in 0:(l - 1)]
    lags_ms = [-m*τ for m in 0:(m - 1)]
    
    lags = Lags([lags_ks; lags_ls; lags_ms])
    
    v = TEVars(Tf = 1:k |> collect, Tpp = 1+(k):l+(k) |> collect, Spp = 1+(l+k):m+(l+k) |> collect)
    customembed(Dataset(source, target), pos, lags), v
end

"""
    te_embed(source, target, cond, k = 1, l = 1, m = 1, n = 1; η = 1, τ = 1)

Create an appropriate custom delay reconstruction from `source`, `target` and `cond` for the
purpose of computing transfer entropy from `source` to `target` conditioned on `cond`. 

# Delay reconstruction

Denote the entries in the source time series by `s(t)`, the entries of the target time 
series by `t(t)` and the entries of the conditional time series by `c(t)`. Then the embedding 
vectors are on the the form

``(T_{f}, T_{pp}, S_{pp}, C_{pp})`` where

``T_f = (t(t + k*η), ..., t(t + 2*η), t(t + η))``,
``T_pp = (t(t), t(t - τ), t(t - 2*τ), ..., t(t - lτ))``,
``S_pp = (s(t), s(t - τ), s(t - 2*τ), ..., s(t - mτ))``,
``C_pp = (c(t), c(t - τ), c(t - 2*τ), ..., c(t - mτ))`` 

and the total dimension of the reconstructed space is `k + l + m + n`.

# Arguments 

- **`source`**: A source data series. 
- **`target`**: A target data series. 
- **`cond`**: A data series on which to condition.
- **`k`**: The dimension of the ``T_f`` component of the delay reconstruction.
- **`l`**: The dimension of the ``T_{pp}`` component of the delay reconstruction.
- **`m`**: The dimension of the ``S_{pp}`` component of the delay reconstruction.
- **`n`**: The dimension of the ``C_{pp}`` component of the delay reconstruction.

# Keyword arguments 
- **`η`**: The prediction lag that goes into the ``T_f`` component of the 
    delay reconstruction.
- **`τ`**: The embedding lag that goes into the ``T_{pp}``, ``S_{pp}`` and ``C_{pp}``
    components of the delay reconstruction.

# Examples 

```julia 
x, y, z = rand(100), rand(100), rand(100)

# Get embedding points and instructions for computing marginals.


# Conditional TE(x -> y | z)
# ==========================
pts, vars = te_embed(x, y, z) # default k = l = m = n = 1
pts, vars = te_embed(x, y, z, η = 2, τ = 3) # specify prediction and embedding lag
pts, vars = te_embed(x, y, z, 2, 1, 2, 1) # specify k = 2, l = 1, m = 2, n = 1
pts, vars = te_embed(x, y, z, 1, 2, 2, 1, η = 2, τ = 3) # specify k = 1, l = 2, m = 2, n = 1

# Compute TE 
# ======================
transferentropy(pts, vars, RectangularBinning(5), VisitationFrequency())
```
"""
function te_embed(source::AbstractArray{T, 1}, 
            target::AbstractArray{T, 1},
            cond::AbstractArray{T, 1},
            k = 1, l = 1, m = 1, n = 1; η = 1, τ = 1) where T

    length(source) == length(target) || throw(ArgumentError("length(source) != length(target)"))
    k >= 1 || throw(ArgumentError("k=$k must be >= 1 (so that Tf has entries)"))
    l >= 1 || throw(ArgumentError("l=$l must be >= 1 (so that Tpp has entries)"))
    m >= 1 || throw(ArgumentError("m=$m must be >= 1 (so that Spp has entries)"))
    n >= 1 || throw(ArgumentError("n=$n must be >= 1 (so that Cpp has entries)"))

    k + l + m + n >= 4 || throw(ArgumentError("k + l + m + n must be at least 4 for meaningful TE computations"))
    
    # Positions (ks and ls both refer to the target time series, 
    # while ms refer to the source time series).
    pos_ks = repeat([2], k) 
    pos_ls = repeat([2], l)
    pos_ms = repeat([1], m)
    pos_ns = repeat([3], n)

    pos = Positions([pos_ks; pos_ls; pos_ms; pos_ns])
    
    # Lags
    lags_ks = [k*η for k in k:-1:1]
    lags_ls = [-l*τ for l in 0:(l - 1)]
    lags_ms = [-m*τ for m in 0:(m - 1)]
    lags_ns = [-n*τ for n in 0:(n - 1)]

    lags = Lags([lags_ks; lags_ls; lags_ms; lags_ns])
    
    v = TEVars(
        Tf = 1:k |> collect, 
        Tpp = 1+(k):l+(k) |> collect, 
        Spp = 1+(l+k):m+(l+k) |> collect,
        Cpp = 1+(l+k+m):n+(l+k+m) |> collect)
    
    customembed(Dataset(source, target, cond), pos, lags), v
end

export te_embed