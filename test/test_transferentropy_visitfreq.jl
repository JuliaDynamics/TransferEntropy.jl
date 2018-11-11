# Estimate transfer entropy from scratch from a random
# set of points `n_realizations` times. If everything
# works as expected, there should be no negative transfer
# entropy values.
ts_length = 100
estimates_3D = Vector{Float64}(undef, n_realizations)
estimates_3D_norm = Vector{Float64}(undef, n_realizations)

@testset "3D #$i" for i in 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:3])
	ϵ = [2, 4, 5]
	v = TEVars([1], [2], [3], Int[])
	estimates_3D[i] = transferentropy_visitfreq(E, ϵ, v)
	estimates_3D_norm[i] = transferentropy_visitfreq(E, ϵ, v, true)
	@test estimates_3D[i] >= 0
	@test estimates_3D_norm[i] >= 0
end

estimates_4D = Vector{Float64}(undef, n_realizations)
estimates_4D_norm = Vector{Float64}(undef, n_realizations)

@testset "4D #$i" for i in 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:4])
	ϵ = 3
	v = TEVars([1], [2], [3, 4], Int[])
	estimates_4D[i] = transferentropy_visitfreq(E, ϵ, v)
	estimates_4D_norm[i] = transferentropy_visitfreq(E, ϵ, v, true)
	@test estimates_4D[i] >= 0
	@test estimates_4D_norm[i] >= 0
end

estimates_5D = Vector{Float64}(undef, n_realizations)
estimates_5D_norm = Vector{Float64}(undef, n_realizations)

@testset "5D #$i" for i in 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:5])
	ϵ = 0.3
	v = TEVars([1], [2], [3, 4], [5])
	estimates_5D[i] = transferentropy_visitfreq(E, ϵ, v)
	estimates_5D_norm[i] = transferentropy_visitfreq(E, ϵ, v, true)
	@test estimates_5D[i] >= 0
	@test estimates_5D_norm[i] >= 0
end
#
# @show estimates_3D
# @show estimates_4D
# @show estimates_5D
# @show estimates_3D_norm
# @show estimates_4D_norm
# @show estimates_5D_norm
# @test all(estimates_3D .>= 0)
# @test all(estimates_4D .>= 0)
# @test all(estimates_5D .>= 0)
#
# @test all(estimates_3D_norm .>= 0)
# @test all(estimates_4D_norm .>= 0)
# @test all(estimates_5D_norm .>= 0)
