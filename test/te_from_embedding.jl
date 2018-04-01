
using Simplices, SimplexSplitting, InvariantDistribution, TransferEntropy
@testset "TE from embedding" begin
	 # Bogus example to trigger compilation
	e_ex = SimplexSplitting.Embedding(
	                InvariantDistribution.invariant_gaussian_embedding(npts = 16))

	# Discrete approximation
	te2 = TransferEntropy.te_from_embedding(e_ex.embedding, 1, n_reps = 2,
								discrete = true, sample_uniformly = true,
								n_randpts = 10)
end
