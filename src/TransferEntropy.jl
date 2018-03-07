__precompile__()
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
include("te_from_triangulation.jl")
include("te_binsizes_ex.jl")
include("te_correlated_gaussians.jl")
include("te_from_triangulation_with_uncertainty.jl")
include("te_from_timeseries.jl")
include("TE_timeseries.jl")
include("TE_examples.jl")

#te_from_timeseries_parallel(npts = 12, covariance = 0.5, parallel = true, sparse = true)

export indexin_rows,
        get_nonempty_bins,
        get_nonempty_bins_abs,
        jointdist,
        marginaldists,
        TEresult,
        embed_correlated_gaussians,
        te_correlated_gaussians, te_correlated_gaussians_init,
        te_from_triangulation,
        te_from_triangulation_withuncertainty,
        te_from_timeseries,
        te_from_timeseries_parallel,
        te_binsizes_ex,
        TE_timeseries,
        TE_timeseries_withuncertainty,
        Examples
end # module
