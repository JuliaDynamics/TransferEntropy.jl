
using Simplices, SimplexSplitting, InvariantDistribution, TransferEntropy
@testset "TE from embedding" begin
	 # Bogus example to trigger compilation
	e_ex = SimplexSplitting.Embedding(
	                InvariantDistribution.invariant_gaussian_embedding(npts = 16))

	# Discrete approximation
	nreps = 2
	te = TransferEntropy.te_from_embedding(e_ex.embedding, 1, n_reps = nreps,
								discrete = true, sample_uniformly = true,
								n_randpts = 10)
	te2 = TransferEntropy.te_from_embedding(e_ex.embedding, 1, n_reps = nreps,
								discrete = true, sample_uniformly = false,
								n_randpts = 10)
	@test typeof(te) == TEresult
	@test size(te.TE, 2) == nreps
	@test typeof(te2) == TEresult
	@test size(te2.TE, 2) == nreps
end
