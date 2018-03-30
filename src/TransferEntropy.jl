__precompile__(true)
module TransferEntropy

using Distributions
using SimplexSplitting
using InvariantDistribution
using TransferEntropy
using ChaoticMaps
@everywhere using Distributions
@everywhere using SimplexSplitting
@everywhere using InvariantDistribution
@everywhere using TransferEntropy
@everywhere using ChaoticMaps
@everywhere using ChaoticMaps.Logistic

@everywhere include("rowindexin.jl")
@everywhere include("get_nonempty_bins.jl")
@everywhere include("joint.jl")
@everywhere include("marginal.jl")
@everywhere include("TEresult.jl")
@everywhere include("te_from_embedding.jl")
@everywhere include("te_from_triang.jl")
@everywhere include("te_from_ts.jl")
@everywhere include("TE_examples.jl")

export indexin_rows,
        get_nonempty_bins,
        get_nonempty_bins_abs,
        jointdist,
        marginaldists,
        TEresult,
        te_from_embedding,
        te_from_triang,
        te_from_ts
        Examples
end # module
