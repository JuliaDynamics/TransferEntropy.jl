using CausalityToolsBase
using DelayEmbeddings 

@test te_embed(rand(100), rand(100)) isa Tuple{CustomReconstruction{3, Float64}, TEVars}
@test te_embed(rand(100), rand(100), 1, 1, 2) isa Tuple{CustomReconstruction{4, Float64}, TEVars}
@test te_embed(rand(100), rand(100), η = 2) isa Tuple{CustomReconstruction{3, Float64}, TEVars}
@test te_embed(rand(100), rand(100), τ = 1) isa Tuple{CustomReconstruction{3, Float64}, TEVars}

@test_throws ArgumentError te_embed(rand(120), rand(100)) isa Tuple{CustomReconstruction{3, Float64}, TEVars}
@test_throws ArgumentError te_embed(rand(100), rand(100), 1, 0, 1) isa Tuple{CustomReconstruction{3, Float64}, TEVars}
@test_throws ArgumentError te_embed(rand(100), rand(100), 0, 1, 1) isa Tuple{CustomReconstruction{3, Float64}, TEVars}
@test_throws ArgumentError te_embed(rand(100), rand(100), 2, 1, 0) isa Tuple{CustomReconstruction{3, Float64}, TEVars}


@test te_embed(rand(100), rand(100), rand(100)) isa Tuple{CustomReconstruction{4, Float64}, TEVars}
@test te_embed(rand(100), rand(100), rand(100), 1, 1, 2, 1) isa Tuple{CustomReconstruction{5, Float64}, TEVars}
@test te_embed(rand(100), rand(100), rand(100), η = 2) isa Tuple{CustomReconstruction{4, Float64}, TEVars}
@test te_embed(rand(100), rand(100), rand(100), τ = 1) isa Tuple{CustomReconstruction{4, Float64}, TEVars}

@test_throws ArgumentError te_embed(rand(120), rand(100), rand(100)) isa Tuple{CustomReconstruction{4, Float64}, TEVars}
@test_throws ArgumentError te_embed(rand(100), rand(100), rand(100), 1, 0, 1, 1) isa Tuple{CustomReconstruction{4, Float64}, TEVars}
@test_throws ArgumentError te_embed(rand(100), rand(100), rand(100), 0, 1, 1, 1) isa Tuple{CustomReconstruction{4, Float64}, TEVars}
@test_throws ArgumentError te_embed(rand(100), rand(100), rand(100), 2, 1, 0, 1) isa Tuple{CustomReconstruction{4, Float64}, TEVars}
