__precompile__(true)

module TransferEntropy

using PerronFrobenius
using Distributions
using GroupSlices
#using ProgressMeter, PmapProgressMeter
#using ChaoticMaps


"""
   TEVars(target_future::Vector{Int}
           target_presentpast::Vector{Int}
           source_presentpast::Vector{Int}
           conditioned_presentpast::Vector{Int})

Which axes of the state space correspond to the future of the target,
the present/past of the target, the present/past of the source, and
the present/past of any conditioned variables? Indices correspond
to column indices of the embedding.points `Dataset`.

This information used by `transferentropy` to compute correct marginal
distributions.
"""
struct TEVars
    target_future::Vector{Int}
    target_presentpast::Vector{Int}
    source_presentpast::Vector{Int}
    conditioned_presentpast::Vector{Int}
end
#include("rowindexin.jl")
#include("get_nonempty_bins.jl")
#include("joint.jl")
#include("marginal.jl")
#include("embed_for_te.jl")
#include("TEresult.jl")
#include("te_from_embedding.jl")
#include("te_from_triang.jl")
#include("te_from_ts.jl")
#include("TE_examples.jl")
include("marginal.jl")
include("te_from_equidistant_binning.jl")

#@everywhere using Distributions
#@everywhere using Simplices, SimplexSplitting, StateSpaceReconstruction
#@everywhere using PerronFrobenius
#@everywhere using ChaoticMaps
#@everywhere using ChaoticMaps.Logistic
#@everywhere using ProgressMeter, PmapProgressMeter
#
#@everywhere include("rowindexin.jl")
#@everywhere include("get_nonempty_bins.jl")
#@everywhere include("joint.jl")
#@everywhere include("marginal.jl")

export indexin_rows,
        # embed_for_te,
	       # get_nonempty_bins, get_nonempty_bins_abs,
        # jointdist, marginaldists,
        # TEresult,
        # Examples,
        #
        # # From equidistant binning
        #marginal, nat_entropy, marginal_multiplicity,
        nat_entropy,
        #
        # # Transfer entropy function (works with many different inputs)
        transferentropy,
        #
        # # Keeping track of which variables goes into which marginals
        TransferEntropyVariables,
        TEVars

end # module
