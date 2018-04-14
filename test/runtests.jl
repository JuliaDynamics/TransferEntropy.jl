using TransferEntropy
using Base.Test

# write your own tests here
#include("random_centroids.jl")



@testset "Transfer entropy" begin

	x = rand(50)
	y = rand(50)
	embedding1 = embed_for_te(x, y, 1)
	embedding2 = embed_for_te(x, y, -1)

	@test !isempty(embedding1)
	@test !isempty(embedding2)

	te = te_from_ts(x, y, n_reps = 1)
	@test length(te) == 5


end


include("te_from_embedding.jl")
