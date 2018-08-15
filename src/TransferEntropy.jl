__precompile__(true)

module TransferEntropy

using Distributions
using GroupSlices
using ProgressMeter, PmapProgressMeter
using PerronFrobenius
using ChaoticMaps

@everywhere using Distributions
@everywhere using Simplices, SimplexSplitting, StateSpaceReconstruction
@everywhere using PerronFrobenius
@everywhere using ChaoticMaps
@everywhere using ChaoticMaps.Logistic
@everywhere using ProgressMeter, PmapProgressMeter
#
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
include("te_from_equidistant_binning.jl")
#
export indexin_rows,
        embed_for_te,
	get_nonempty_bins, get_nonempty_bins_abs,
        jointdist, marginaldists,
        te_from_embedding,
        te_from_equidistant_binning,
        te_from_triang, te_from_triang_multiple_binsizes,
        te_from_ts,
        TEresult,
        Examples,

        # From equidistant binning
        marginal, nat_entropy, marginal_multiplicity,
        transferentropy, shape_transferentropy,

        # Keeping track of which variables goes into which marginals
        TransferEntropyVariables

end # module
