module Examples

export embed_for_te,
    invariant_logistic

using ChaoticMaps
using SimplexSplitting
using InvariantDistribution

function invariant_logistic(n_pts::Int, lag::Int)

    attempts = 0
    while attempts <= 100
        attempts += 1

        l = ChaoticMaps.Logistic.logistic_map(n_pts = n_pts)
        if isa(l, Float64)
            success = false

            while !success
                l = ChaoticMaps.Logistic.logistic_map(n_pts = n_pts)
                if !isa(l, Float64)
                    success = true
                end
            end
        end

        embedding = invariantize_embedding(embed_for_te(l.X, l.Y, lag))
        if !isempty(embedding)
            return l
        end
    end

    return Float64[]
end

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
    lag::Int = 1
    )

    return hcat(source[1:end-lag], target[1:end-lag], target[1+lag:end])
end

end
