addprocs(4)

@everywhere using Simplices, SimplexSplitting, InvariantDistribution, TransferEntropy

 # Bogus example to trigger compilation
e_ex = SimplexSplitting.Embedding(InvariantDistribution.invariant_gaussian_embedding(npts = 12))
t_ex = SimplexSplitting.triang_from_embedding(e_ex)
mm = InvariantDistribution.mm_discrete_dense(t_ex)
@assert all(sum(mm, 2) .≈ 1)

te = TransferEntropy.te_from_embedding(e_ex.embedding, 1, n_reps = 200)
te2 = TransferEntropy.te_from_embedding(e_ex.embedding, 1, n_reps = 100)
plot(te.binsizes, median(te.TE, 2), color = "black")
plot!(te2.binsizes, median(te2.TE, 2), color = "red")



# using Plots; plotlyjs()
using ChaoticMaps.Rossler
u0 = rand(6)
rp = RosslerCoupledParams(ω₁ = 1-0.015, ω₂ = 1+0.015, ϵ₂=0.1)
s1 = solve_rossler_coupled(rp = rp, tend = 500, timestep = 2, u0 = u0)
# using ChaoticMaps.Rossler
# p1 = plot(s1, vars=(1, 2, 3), color = "black", width = 1)
# plot!(p1, s1, vars = (4,5,6), color = "blue", width = 1)
#
# p2 = plot(s1, vars=(0, 1), color = "black", width = 1, denseplot = false)
# plot!(p2, s1, vars=(0, 4), color = "blue", width = 1, denseplot = false)
#


sol1 = Array(s1)
X = vec(sol1[1, 100:end])
Y = vec(sol1[4, 100:end])
te_lag = 1
# Triangulate
embedding = hcat(X[1:end-te_lag],
                Y[1:end-te_lag],
                Y[(1 + te_lag):end])


embedding = InvariantDistribution.invariantize_embedding(embedding, max_point_remove = 10)
t = InvariantDistribution.triang_from_embedding(SimplexSplitting.Embedding(embedding))
te_rossler = TransferEntropy.te_from_embedding(embedding, 1)
