module TransferEntropy

include("rowindexin.jl")
include("get_nonempty_bins.jl")
include("joint.jl")
include("marginal.jl")
include("te_from_triangulation.jl")
include("te_binsizes_ex.jl")
include("te_correlated_gaussians.jl")

export indexin_rows,
        get_nonempty_bins, get_nonempty_bins_abs,
        marginaldists,
        jointdist,
        embed_correlated_gaussians,
        te_correlated_gaussians, te_correlated_gaussians_init,
        te_from_triangulation,
        te_binsizes_ex

end # module
