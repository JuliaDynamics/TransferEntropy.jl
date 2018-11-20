# Estimate transfer entropy from scratch from a random
# set of points `n_realizations` times.
ts_length = 100

# Estimate transfer entropy using the wrapper function,
# and manually (doing the steps performed inside the
# wrapper function) to make sure that they give the same
# result.
estimates_3D_wrapper = Vector{Float64}(undef, n_realizations)
estimates_3D_allsteps = Vector{Float64}(undef, n_realizations)
estimates_3D_wrapper_norm = Vector{Float64}(undef, n_realizations)
estimates_3D_allsteps_norm = Vector{Float64}(undef, n_realizations)

@testset "3D #$i" for i in 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:3])
	ϵ = 3
	# Test by doing all the dirty work and providing the raw input to the estimator
	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = organize_bin_labels(bins_visited_by_orbit)
	TO = transferoperator_binvisits(bininfo)
	iv = left_eigenvector(TO)
	v = TEVars([1], [2], [3], Int[])

	estimates_3D_wrapper[i] = tetogrid(E, ϵ, v)
	estimates_3D_allsteps[i] = tetogrid(bins_visited_by_orbit, iv, v)
	estimates_3D_wrapper_norm[i] = tetogrid(E, ϵ, v; normalise_to_tPP = true)
	estimates_3D_allsteps_norm[i] = tetogrid(bins_visited_by_orbit, iv, v; normalise_to_tPP = true)
	@test estimates_3D_wrapper[i] >= 0
	@test estimates_3D_allsteps[i] >= 0
	@test estimates_3D_wrapper_norm[i] >= 0
	@test estimates_3D_allsteps_norm[i] >= 0
end

ts_length = 200
estimates_4D_wrapper = Vector{Float64}(undef, n_realizations)
estimates_4D_allsteps = Vector{Float64}(undef, n_realizations)
estimates_4D_wrapper_norm = Vector{Float64}(undef, n_realizations)
estimates_4D_allsteps_norm = Vector{Float64}(undef, n_realizations)

@testset "4D #$i" for i in 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:4])
	ϵ = 0.3
	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = organize_bin_labels(bins_visited_by_orbit)
	TO = transferoperator_binvisits(bininfo)
	iv = left_eigenvector(TO)
	v = TEVars([1], [2], [3, 4], Int[])
	estimates_4D_wrapper[i] = tetogrid(E, ϵ, v)
	estimates_4D_allsteps[i] = tetogrid(bins_visited_by_orbit, iv, v)
	estimates_4D_wrapper_norm[i] = tetogrid(E, ϵ, v; normalise_to_tPP = true)
	estimates_4D_allsteps_norm[i] = tetogrid(bins_visited_by_orbit, iv, v; normalise_to_tPP = true)
	@test estimates_4D_wrapper[i] >= 0
	@test estimates_4D_allsteps[i] >= 0
	@test estimates_4D_wrapper_norm[i] >= 0
	@test estimates_4D_allsteps_norm[i] >= 0
end

ts_length = 300
estimates_5D_wrapper = Vector{Float64}(undef, n_realizations)
estimates_5D_allsteps = Vector{Float64}(undef, n_realizations)
estimates_5D_wrapper_norm = Vector{Float64}(undef, n_realizations)
estimates_5D_allsteps_norm = Vector{Float64}(undef, n_realizations)

@testset "5D #$i" for i in 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:5])
	ϵ = [0.2, 0.2, 0.1, 0.2, 0.3]

	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = organize_bin_labels(bins_visited_by_orbit)
	TO = transferoperator_binvisits(bininfo)
	iv = left_eigenvector(TO)
	v = TEVars([1], [2], [3, 4], [5])
	estimates_5D_wrapper[i] = tetogrid(E, ϵ, v)
	estimates_5D_allsteps[i] = tetogrid(bins_visited_by_orbit, iv, v)
	estimates_5D_wrapper_norm[i] = tetogrid(E, ϵ, v; normalise_to_tPP = true)
	estimates_5D_allsteps_norm[i] = tetogrid(bins_visited_by_orbit, iv, v; normalise_to_tPP = true)
	@test estimates_5D_wrapper[i] >= 0
	@test estimates_5D_allsteps[i] >= 0
	@test estimates_5D_wrapper_norm[i] >= 0
	@test estimates_5D_allsteps_norm[i] >= 0
end

# # If everything  works as expected, there should be no negative
# # transfer entropy values.
#@show estimates_3D_wrapper
#@show estimates_4D_wrapper
#@show estimates_5D_wrapper
#show estimates_3D_allsteps
#@show estimates_4D_allsteps
#@show estimates_5D_allsteps

#@test all(estimates_3D_wrapper .>= 0)
#@test all(estimates_4D_wrapper .>= 0)
#@test all(estimates_5D_wrapper .>= 0)
#@test all(estimates_3D_allsteps .>= 0)
#@test all(estimates_4D_allsteps .>= 0)
#@test all(estimates_5D_allsteps .>= 0)

#@show estimates_3D_wrapper_norm
#@show estimates_4D_wrapper_norm
#@show estimates_5D_wrapper_norm
#@show estimates_3D_allsteps_norm
#@show estimates_4D_allsteps_norm
#@show estimates_5D_allsteps_norm

#@test all(estimates_3D_wrapper_norm .>= 0)
#@test all(estimates_4D_wrapper_norm .>= 0)
#@test all(estimates_5D_wrapper_norm .>= 0)
#@test all(estimates_3D_allsteps_norm .>= 0)
#@test all(estimates_4D_allsteps_norm .>= 0)
#@test all(estimates_5D_allsteps_norm .>= 0)

# The invariant distribution is estimated independently in the calls
# to the different transfer entropy estimators. The distribution is
# estimated by repeated application of the transfer operator on a
# randomly initialised distribution until convergence is achieved.
# Because a different initial distribution is used for each function
# call, the estimated transfer entropy will be slightly different.
# However, we do not expect the differences to be very large.
#@test all(abs.(estimates_3D_wrapper .- estimates_3D_allsteps) .< 1e-2)
#@test all(abs.(estimates_4D_wrapper .- estimates_4D_allsteps) .< 1e-2)
#@test all(abs.(estimates_5D_wrapper .- estimates_5D_allsteps) .< 1e-2)
