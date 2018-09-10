using TransferEntropy
using Base.Test

@testset "Estimators" begin
	@time include("test_transferentropy_transferoperator.jl")
	@time include("test_transferentropy_visitfreq.jl")
end
