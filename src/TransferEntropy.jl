__precompile__(true)

module TransferEntropy

using Reexport
@reexport using PerronFrobenius
@reexport using StateSpaceReconstruction
using Distributions
using GroupSlices


"""
   TEVars(
      target_future::Vector{Int}
      target_presentpast::Vector{Int}
      source_presentpast::Vector{Int}
      conditioned_presentpast::Vector{Int}
   )

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

include("entropy.jl")
include("estimators/transferentropy_visitfreq.jl")
#include("TEresult.jl")


export
entropy,
TEVars,
#@everywhere using ChaoticMaps
transferentropy_visitfreq
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
