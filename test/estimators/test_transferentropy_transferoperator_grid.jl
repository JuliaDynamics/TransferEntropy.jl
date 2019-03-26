import StateSpaceReconstruction:
	cembed,
	assign_bin_labels

import PerronFrobenius:
	get_binvisits,
	estimate_transferoperator_from_binvisits,
	invariantmeasure
import CausalityToolsBase: encode, RectangularBinning, get_minima_and_edgelengths


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
	E = cembed([diff(rand(ts_length)) for i = 1:3])
	ϵ = 3
	# Test by doing all the dirty work and providing the raw input to the estimator
	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = get_binvisits(bins_visited_by_orbit)
	TO = estimate_transferoperator_from_binvisits(bininfo)
	iv = invariantmeasure(TO)
	v = TEVars([1], [2], [3], Int[])

	estimates_3D_wrapper[i] = tetogrid(E, ϵ, v)
	estimates_3D_allsteps[i] = tetogrid(bins_visited_by_orbit, iv, v)
	estimates_3D_wrapper_norm[i] = tetogrid(E, ϵ, v)
	estimates_3D_allsteps_norm[i] = tetogrid(bins_visited_by_orbit, iv, v)
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
	E = cembed([diff(rand(ts_length)) for i = 1:4])
	ϵ = 0.3
	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = get_binvisits(bins_visited_by_orbit)
	TO = estimate_transferoperator_from_binvisits(bininfo)
	iv = invariantmeasure(TO)
	v = TEVars([1], [2], [3, 4], Int[])
	estimates_4D_wrapper[i] = tetogrid(E, ϵ, v)
	estimates_4D_allsteps[i] = tetogrid(bins_visited_by_orbit, iv, v)
	estimates_4D_wrapper_norm[i] = tetogrid(E, ϵ, v)
	estimates_4D_allsteps_norm[i] = tetogrid(bins_visited_by_orbit, iv, v)
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
	E = cembed([diff(rand(ts_length)) for i = 1:5])
	ϵ = [0.2, 0.2, 0.1, 0.2, 0.3]

	bins_visited_by_orbit = assign_bin_labels(E, ϵ)
	bininfo = get_binvisits(bins_visited_by_orbit)
	TO = estimate_transferoperator_from_binvisits(bininfo)
	iv = invariantmeasure(TO)
	v = TEVars([1], [2], [3, 4], [5])
	estimates_5D_wrapper[i] = tetogrid(E, ϵ, v)
	estimates_5D_allsteps[i] = tetogrid(bins_visited_by_orbit, iv, v)
	estimates_5D_wrapper_norm[i] = tetogrid(E, ϵ, v)
	estimates_5D_allsteps_norm[i] = tetogrid(bins_visited_by_orbit, iv, v)
	@test estimates_5D_wrapper[i] >= 0
	@test estimates_5D_allsteps[i] >= 0
	@test estimates_5D_wrapper_norm[i] >= 0
	@test estimates_5D_allsteps_norm[i] >= 0
end













using StaticArrays
using DelayEmbeddings
PTS = [[rand(3) for i = 1:50] for k = 1:n_realizations]

@testset "3D #$i" for i in 1:n_realizations
	pts = PTS[i]
	spts = [SVector{3, Float64}(pt) for pt in PTS[i]]
	mpts = [MVector{3, Float64}(pt) for pt in PTS[i]]
	D = Dataset(pts)

	ϵ = 3
	# Test by doing all the dirty work and providing the raw input to the estimator
	mini, edgelengths = get_minima_and_edgelengths(pts, ϵ)
	encoded_pts = hcat(encode(pts, mini, edgelengths)...,)
	encoded_spts = hcat(encode(spts, mini, edgelengths)...,)
	encoded_mpts = hcat(encode(mpts, mini, edgelengths)...,)
	encoded_D = hcat(encode(D, mini, edgelengths)...,)

	binvisits_pts = get_binvisits(encoded_pts)
	binvisits_spts = get_binvisits(encoded_spts)
	binvisits_mpts = get_binvisits(encoded_mpts)
	binvisits_D = get_binvisits(encoded_D)

	# Approximate the transfer operator
	TO_pts = estimate_transferoperator_from_binvisits(binvisits_pts)
	TO_spts = estimate_transferoperator_from_binvisits(binvisits_spts)
	TO_mpts = estimate_transferoperator_from_binvisits(binvisits_mpts)
	TO_D = estimate_transferoperator_from_binvisits(binvisits_D)

	# Get the invariant measure from the transfer operator
	iv_pts = invariantmeasure(TO_pts)
	iv_spts = invariantmeasure(TO_spts)
	iv_mpts = invariantmeasure(TO_mpts)
	iv_D = invariantmeasure(TO_D)

	v = TEVars([1], [2], [3], Int[])

	@test tetogrid(encoded_pts, iv_pts, v) >= 0
	@show tetogrid(encoded_spts, iv_spts, v) >= 0
	@show tetogrid(encoded_mpts, iv_mpts, v) >= 0
	@show tetogrid(encoded_D, iv_D, v) >= 0

	#estimates_3D_wrapper[i] = tetogrid(E, ϵ, v)
	#estimates_3D_allsteps[i] = tetogrid(bins_visited_by_orbit, iv, v)
	#estimates_3D_wrapper_norm[i] = tetogrid(E, ϵ, v)
	#estimates_3D_allsteps_norm[i] = tetogrid(bins_visited_by_orbit, iv, v)
	#@test estimates_3D_wrapper[i] >= 0
	#@test estimates_3D_allsteps[i] >= 0
	#@test estimates_3D_wrapper_norm[i] >= 0
	#@test estimates_3D_allsteps_norm[i] >= 0
end
