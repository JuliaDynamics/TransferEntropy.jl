using TransferEntropy
using Base.Test
using Distances

@testset "Estimators" begin
	@time include("test_transferentropy_transferoperator.jl")
	@time include("test_transferentropy_visitfreq.jl")
	@time include("test_transferentropy_kraskov.jl")
end
