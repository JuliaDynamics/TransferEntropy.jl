module Examples

export embed_for_te,
    invariant_logistic,
    generate_ensemble_invariantlogistic

using ChaoticMaps
using SimplexSplitting
using Distributions
using InvariantDistribution

"""
    embed_for_te(
        source::Vector{Float64},
        target::Vector{Float64},
        lag::Int
        )

Embed a `source` time series and a `target` time series for transfer entropy analysis. The
lag appearing in the transfer entropy computation is given by `lag`.
"""
function embed_for_te(
    source::Vector{Float64},
    target::Vector{Float64},
    lag::Int
    )
    if lag > 0
        return hcat(source[1:end-lag], target[1:end-lag], target[1+lag:end])
    else lag < 0
        return hcat(source[1+abs(lag):end], target[1+abs(lag):end], target[1:end-abs(lag)])
    end
end

"""
Generate `n_samples` samples of coupled logistic map time series (with components X and Y)
that have been verified to form invariant sets when embedded for TE using the given
embedding `lag`.
"""
function invariant_logistic(
        n_pts::Int, lag::Int;
        μXdist::Distributions.Distribution = Uniform(2, 4),
        μYdist::Distributions.Distribution = Uniform(3.7, 3.9),
        initdist::Distributions.Distribution = Uniform(0.1, 0.9),
        αdist::Distributions.Distribution = Uniform(0.05, 0.1),
        x₀::Float64 = rand(initdist),
        y₀::Float64 = rand(initdist),
        μX::Float64 = rand(μXdist),
        μY::Float64 = rand(μYdist),
        αXonY::Float64 = rand(αdist),
        αYonX::Float64 = rand(αdist),
        n_transient::Int = 10000,
        stepsize::Int = 1,
        noise_frac::Float64 = 0.0,
        seed::Array{Int32,1} = reinterpret(Int32, Base.Random.GLOBAL_RNG.seed))


    params = ChaoticMaps.Logistic.TwoSpecies.Params(
        μXdist = μXdist,
        μYdist = μYdist,
        initdist = initdist,
        αdist = αdist,
        x₀ = x₀,
        y₀ = y₀,
        μX = μX,
        μY = μY,
        αXonY = αXonY,
        αYonX = αYonX)

    info = ChaoticMaps.Info.MapInfo(n_pts = n_pts,
        n_transient = n_transient,
        stepsize = stepsize,
        noise_frac = noise_frac,
        seed = seed)

    lm_twospecies = ChaoticMaps.Logistic.TwoSpecies.itermap(p = params, info = info)

    embedding = InvariantDistribution.invariantize_embedding(
        embed_for_te(lm_twospecies.X, lm_twospecies.Y, lag)
    )

    if typeof(embedding) == Void # if embedding encounters errors
        warn("Trying again with different time series...")
        return invariant_logistic(n_pts, lag)
    else
        return lm_twospecies
    end
end

"""
Almost the same as above.

Generate `n_samples` samples of coupled logistic map time series (with components X and Y)
that have been verified to form invariant sets when embedded for TE using multiple
embedding `lags`.
"""
function invariant_logistic(n_pts::Int, lags::Range;
        μXdist::Distributions.Distribution = Uniform(2, 4),
        μYdist::Distributions.Distribution = Uniform(3.7, 3.9),
        initdist::Distributions.Distribution = Uniform(0.1, 0.9),
        αdist::Distributions.Distribution = Uniform(0.05, 0.1),
        x₀::Float64 = rand(initdist),
        y₀::Float64 = rand(initdist),
        μX::Float64 = rand(μXdist),
        μY::Float64 = rand(μYdist),
        αXonY::Float64 = rand(αdist),
        αYonX::Float64 = rand(αdist),
        n_transient::Int = 10000,
        stepsize::Int = 1,
        noise_frac::Float64 = 0.0,
        seed::Array{Int32,1} = reinterpret(Int32, Base.Random.GLOBAL_RNG.seed))


    params = ChaoticMaps.Logistic.TwoSpecies.Params(
        μXdist = μXdist,
        μYdist = μYdist,
        initdist = initdist,
        αdist = αdist,
        x₀ = x₀,
        y₀ = y₀,
        μX = μX,
        μY = μY,
        αXonY = αXonY,
        αYonX = αYonX)

    info = ChaoticMaps.Info.MapInfo(n_pts = n_pts,
        n_transient = n_transient,
        stepsize = stepsize,
        noise_frac = noise_frac,
        seed = seed)

    lm_twospecies = ChaoticMaps.Logistic.TwoSpecies.itermap(p = params, info = info)

    embeddings = Vector{Any}(length(lags))

    for i = 1:length(lags)
        embeddings[i] = InvariantDistribution.invariantize_embedding(
            embed_for_te(lm_twospecies.X, lm_twospecies.Y, lags[i])
        )
    end

    if any([typeof(embedding) == Void for embedding in embeddings])
        warn("Trying again with different time series...")
        return invariant_logistic(n_pts, lags)
    else
        return lm_twospecies
    end
end


"""
Exactly the same as above, but allowing Vector{Int} as lags

Generate `n_samples` samples of coupled logistic map time series (with components X and Y)
that have been verified to form invariant sets when embedded for TE using multiple
embedding `lags`.
"""
function invariant_logistic(n_pts::Int, lags::Vector{Int};
        μXdist::Distributions.Distribution = Uniform(2, 4),
        μYdist::Distributions.Distribution = Uniform(3.7, 3.9),
        initdist::Distributions.Distribution = Uniform(0.1, 0.9),
        αdist::Distributions.Distribution = Uniform(0.05, 0.1),
        x₀::Float64 = rand(initdist),
        y₀::Float64 = rand(initdist),
        μX::Float64 = rand(μXdist),
        μY::Float64 = rand(μYdist),
        αXonY::Float64 = rand(αdist),
        αYonX::Float64 = rand(αdist),
        n_transient::Int = 10000,
        stepsize::Int = 10,
        noise_frac::Float64 = 0.0,
        seed::Array{Int32,1} = reinterpret(Int32, Base.Random.GLOBAL_RNG.seed))


    params = ChaoticMaps.Logistic.TwoSpecies.Params(
        μXdist = μXdist,
        μYdist = μYdist,
        initdist = initdist,
        αdist = αdist,
        x₀ = x₀,
        y₀ = y₀,
        μX = μX,
        μY = μY,
        αXonY = αXonY,
        αYonX = αYonX)

    info = ChaoticMaps.Info.MapInfo(n_pts = n_pts,
        n_transient = n_transient,
        stepsize = stepsize,
        noise_frac = noise_frac,
        seed = seed)

    lm_twospecies = ChaoticMaps.Logistic.TwoSpecies.itermap(p = params, info = info)

    embeddings = Vector{Any}(length(lags))

    for i = 1:length(lags)
        embeddings[i] = InvariantDistribution.invariantize_embedding(
            embed_for_te(lm_twospecies.X, lm_twospecies.Y, lags[i])
        )
    end

    if any([typeof(embedding) == Void for embedding in embeddings])
        warn("Trying again with different time series...")
        return invariant_logistic(n_pts, lags)
    else
        return lm_twospecies
    end
end

"""
Generate an ensemble of `n_members` logistic maps for all combinations of forcing strengths
``αXonYs` and `αYonXs`. The resulting embeddings are guaranteed to be invariant under the
forward linear map for all lags given in `lags` (zero lag doesn't work and is not included,
because it doesn't provide any directional information).

"""
function generate_ensemble_invariantlogistic(
        n_pts::Int,
        lags::Any,
        n_members::Int = 5;
        αXonYs::Range = 0:0.05:0.5,
        αYonXs::Range = 0:0.05:0.5,
        μXdist::Distributions.Distribution = Uniform(2, 4),
        μYdist::Distributions.Distribution = Uniform(3.7, 3.9),
        initdist::Distributions.Distribution = Uniform(0.1, 0.9),
        αdist::Distributions.Distribution = Uniform(0.05, 0.1),
        x₀::Float64 = rand(initdist),
        y₀::Float64 = rand(initdist),
        μX::Float64 = rand(μXdist),
        μY::Float64 = rand(μYdist),
        n_transient::Int = 10000,
        stepsize::Int = 10,
        noise_frac::Float64 = 0.0,
        seed::Array{Int32,1} = reinterpret(Int32, Base.Random.GLOBAL_RNG.seed))

    ensemble = Array{Any}(length(αXonYs), length(αYonXs), n_members)
    for i = 1:length(αXonYs)
        for j = 1:length(αYonXs)
            for k = 1:n_members
                ensemble[i, j, k] = invariant_logistic(
                    n_pts,
                    lags,
                    μXdist = μXdist,
                    μYdist = μYdist,
                    initdist = initdist,
                    αdist = αdist,
                    x₀ = x₀,
                    y₀ = y₀,
                    μX = μX,
                    μY = μY,
                    αXonY = αXonYs[i],
                    αYonX = αYonXs[j],
                    n_transient = n_transient,
                    stepsize = stepsize,
                    noise_frac = noise_frac,
                    seed = seed)
            end
        end
    end
    return ensemble
end

"""
Perform transfer entropy on correlated gaussian time series of length `n_pts`
whose coupling is dictated by their `covariance` and the time series are
related by a lag of `tau`.
"""
function te_correlated_gaussians(n_pts, covariance; tau = 1)

	gaussian_embedding = InvariantDistribution.invariant_gaussian_embedding(
		npts = n_pts,
		cov = covariance,
		tau = tau
	)

	te_from_embedding(invariantize_embedding(gaussian_embedding))
end


end
