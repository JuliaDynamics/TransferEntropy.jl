if lowercase(get(ENV, "CI", "false")) == "true"
    include("install_test_dependencies.jl")
end

using TransferEntropy
using CausalityToolsBase
using Test
using Distances
using DelayEmbeddings
using StaticArrays

include("test_interface.jl")

# n_realizations = 5

# @testset "Visitation frequency estimator" begin
# 	include("estimators/test_transferentropy_visitfreq.jl")
# end

# @testset "kNN (Kraskov) estimator" begin
# 	include("estimators/test_transferentropy_kraskov.jl")
# end

# @testset "Transfer operator grid estimator" begin
# 	include("estimators/test_transferentropy_transferoperator_grid.jl")
# end

# include("estimators/test_common_interface.jl")
# include("test_te_embed.jl")

# include("test_convenience_funcs_regular_TE.jl")
# include("test_convenience_funcs_conditional_TE.jl")