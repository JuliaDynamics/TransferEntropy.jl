__precompile__(true)

module TransferEntropy

using Reexport
@reexport using PerronFrobenius
@reexport using StateSpaceReconstruction
using Distributions
using GroupSlices
import Simplices.childpoint

"""
    TEVars(target_future::Vector{Int}
        target_presentpast::Vector{Int}
        source_presentpast::Vector{Int}
        conditioned_presentpast::Vector{Int})

Which axes of the state space correspond to the future of the target,
the present/past of the target, the present/past of the source, and
the present/past of any conditioned variables? Indices correspond
to column indices of the embedding.points `Dataset`.

This information is used by the transfer entropy estimators to ensure
the marginal distributions are computed correctly.
"""
struct TEVars
    target_future::Vector{Int}
    target_presentpast::Vector{Int}
    source_presentpast::Vector{Int}
    conditioned_presentpast::Vector{Int}
end

TEVars(target_future::Vector{Int},
    target_presentpast::Vector{Int},
    source_presentpast::Vector{Int}) =
    TEVars(target_future, target_presentpast, source_presentpast, Int[])

include("area_under_curve.jl")
include("entropy.jl")
include("estimators/transferentropy_visitfreq.jl")
include("estimators/transferentropy_transferoperator.jl")
include("estimators/transferentropy_triangulation.jl")

export
entropy,
TEVars,
te_integral, ∫te,
transferentropy_transferoperator_visitfreq, tetofreq,
transferentropy_visitfreq, tefreq,
transferentropy_transferoperator_triang, tetotri

end # module
