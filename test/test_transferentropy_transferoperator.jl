# Estimate transfer entropy from scratch from a random
# set of points `n_realizations` times.
n_realizations = 10
ts_length = 200

# Estimate transfer entropy using the wrapper function,
# and manually (doing the steps performed inside the
# wrapper function) to make sure that they give the same
# result.
estimates_3D_wrapper = Vector{Float64}(n_realizations)
estimates_3D_allsteps = Vector{Float64}(n_realizations)

for i = 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:3])
	ϵ = 3
	# Test by doing all the dirty work and providing the raw input to the estimator
	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = organize_bin_labels(bins_visited_by_orbit)
	TO = transferoperator(bininfo, 1.0)
	iv = left_eigenvector(TO)
	v = TEVars([1], [2], [3], Int[])
	est_allsteps = transferentropy_transferoperator_visitfreq(bins_visited_by_orbit, iv, v)

	# Test by providing an embedding and bin size, and let the estimator
	# take care of the rest.
	est_wrapper = transferentropy_transferoperator_visitfreq(E, ϵ, v)
	estimates_3D_wrapper[i] = est_wrapper
	estimates_3D_allsteps[i] = est_allsteps

end

estimates_4D_wrapper = Vector{Float64}(n_realizations)
estimates_4D_allsteps = Vector{Float64}(n_realizations)
for i = 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:4])
	ϵ = 0.5
	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = organize_bin_labels(bins_visited_by_orbit)
	TO = transferoperator(bininfo, 1.0)
	iv = left_eigenvector(TO)
	v = TEVars([1], [2], [3, 4], Int[])
	est_allsteps = transferentropy_transferoperator_visitfreq(bins_visited_by_orbit, iv, v)
	est_wrapper = transferentropy_transferoperator_visitfreq(E, ϵ, v)
	estimates_4D_wrapper[i] = est_wrapper
	estimates_4D_allsteps[i] = est_allsteps

end

estimates_5D_wrapper = Vector{Float64}(n_realizations)
estimates_5D_allsteps = Vector{Float64}(n_realizations)
for i = 1:n_realizations
	E = embed([diff(rand(ts_length)) for i = 1:5])
	ϵ = [0.1, 0.2, 0.1, 0.2, 0.3]

	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = organize_bin_labels(bins_visited_by_orbit)
	TO = transferoperator(bininfo, 1.0)
	iv = left_eigenvector(TO)
	v = TEVars([1], [2], [3, 4], [5])
	est_allsteps = transferentropy_transferoperator_visitfreq(bins_visited_by_orbit, iv, v)
	est_wrapper = transferentropy_transferoperator_visitfreq(E, ϵ, v)
	estimates_5D_wrapper[i] = est_wrapper
	estimates_5D_allsteps[i] = est_allsteps
end

# # If everything  works as expected, there should be no negative
# # transfer entropy values.
@test all(estimates_3D_wrapper .>= 0)
@test all(estimates_4D_wrapper .>= 0)
@test all(estimates_5D_wrapper .>= 0)

@test all(estimates_3D_allsteps .>= 0)
@test all(estimates_4D_allsteps .>= 0)
@test all(estimates_5D_allsteps .>= 0)

# The invariant distribution is estimated independently in the calls
# to the different transfer entropy estimators. The distribution is
# estimated by repeated application of the transfer operator on a
# randomly initialised distribution until convergence is achieved.
# Because a different initial distribution is used for each function
# call, the estimated transfer entropy will be slightly different.
# However, we do not expect differences larger than 1e-2.
#@test all(abs.(estimates_3D_wrapper .- estimates_3D_allsteps) .< 1e-2)
#@test all(abs.(estimates_4D_wrapper .- estimates_4D_allsteps) .< 1e-2)
#@test all(abs.(estimates_5D_wrapper .- estimates_5D_allsteps) .< 1e-2)
