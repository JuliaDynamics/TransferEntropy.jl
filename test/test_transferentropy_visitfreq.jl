# Estimate transfer entropy from scratch from a random
# set of points `n_realizations` times. If everything
# works as expected, there should be no negative transfer
# entropy values.
n_realizations = 30
ts_length = 500
estimates_3D = Vector{Float64}(n_realizations)
for i = 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:3])
	ϵ = rand(1:5)
	v = TEVars([1], [2], [3], Int[])
	est = tefreq(E, ϵ, v)
	estimates_3D[i] = est
end

estimates_4D = Vector{Float64}(n_realizations)
for i = 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:4])
	ϵ = rand(1:5)
	v = TEVars([1], [2], [3, 4], Int[])
	est = tefreq(E, ϵ, v)
	estimates_4D[i] = est
end

estimates_5D = Vector{Float64}(n_realizations)
for i = 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:5])
	ϵ = rand(1:5)
	v = TEVars([1], [2], [3, 4], [5])
	est = tefreq(E, ϵ, v)
	estimates_5D[i] = est
end

@test all(estimates_3D .>= 0)
@test all(estimates_4D .>= 0)
@test all(estimates_5D .>= 0)
