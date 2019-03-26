if lowercase(get(ENV, "CI", "false")) == "true"
    include("install_dependencies.jl")
end

using TransferEntropy
using CausalityToolsBase
using Test
using Distances
using DelayEmbeddings
using StaticArrays

n_realizations = 30
# Also test old funcs
@testset "Visitation frequency estimator" begin
	include("estimators/test_transferentropy_visitfreq.jl")
end

@testset "kNN (Kraskov) estimator" begin
	include("estimators/test_transferentropy_kraskov.jl")
end

@testset "Transfer operator grid estimator" begin
	include("estimators/test_transferentropy_transferoperator_grid.jl")
end

# Test new interface

include("estimators/test_common_interface.jl")