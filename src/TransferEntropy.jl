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
include("estimators/transferentropy_transferoperator.jl")


export
entropy,
TEVars,
transferentropy_transferoperator,
transferentropy_visitfreq

end # module
