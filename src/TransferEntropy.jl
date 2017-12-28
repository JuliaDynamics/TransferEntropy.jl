module TransferEntropy

include("rowindexin.jl")
include("get_nonempty_bins.jl")
include("joint.jl")
include("marginal.jl")
include("te_from_triangulation.jl")

export indexin_rows, get_nonempty_bins,
        marginaldists, jointdist,
        te_from_triangulation

end # module
