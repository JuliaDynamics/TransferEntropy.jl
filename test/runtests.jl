using TransferEntropy
using Base.Test
using Distances

@testset "Visitation frequency estimator" begin
	include("test_transferentropy_visitfreq.jl")
end

@testset "kNN (Kraskov) estimator" begin
	include("test_transferentropy_kraskov.jl")
end

@testset "Transfer operator, grid approach estimator" begin
	include("test_transferentropy_transferoperator.jl")
end
