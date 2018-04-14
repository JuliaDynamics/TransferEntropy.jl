__precompile__(true)

module TransferEntropy

using Distributions, ProgressMeter, PmapProgressMeter
using Simplices
using SimplexSplitting
using InvariantDistribution
using TransferEntropy
using ChaoticMaps

@everywhere using Distributions
@everywhere using Simplices
@everywhere using SimplexSplitting
@everywhere using InvariantDistribution
@everywhere using TransferEntropy
@everywhere using ChaoticMaps
@everywhere using ChaoticMaps.Logistic
@everywhere using ProgressMeter
@everywhere using PmapProgressMeter

@everywhere include("rowindexin.jl")
@everywhere include("get_nonempty_bins.jl")
@everywhere include("joint.jl")
@everywhere include("marginal.jl")
include("embed_for_te.jl")
include("TEresult.jl")
include("te_from_embedding.jl")
include("te_from_triang.jl")
include("te_from_ts.jl")
include("TE_examples.jl")

export indexin_rows,
        embed_for_te,
	get_nonempty_bins, get_nonempty_bins_abs,
        jointdist, marginaldists,
        te_from_embedding,
        te_from_triang, te_from_triang_multiple_binsizes,
        te_from_ts,
        TEresult,
        Examples

end # module
