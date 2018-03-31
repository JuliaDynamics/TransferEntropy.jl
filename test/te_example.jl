addprocs(2)
@everywhere using TransferEntropy

 # Bogus example to trigger compilation
e_ex = SimplexSplitting.Embedding(InvariantDistribution.invariant_gaussian_embedding(npts = 15))
t_ex = SimplexSplitting.triang_from_embedding(e_ex)
mm = InvariantDistribution.mm_discrete_dense(t_ex)
@assert all(sum(mm, 2) .≈ 1)

using TransferEntropy
te = TransferEntropy.te_from_embedding(e_ex.embedding, 1)

te
using Plots; plotlyjs()
plot(te.binsizes, te.TE)
