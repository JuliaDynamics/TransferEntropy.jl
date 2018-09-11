using Distances
using NearestNeighbors

# Some random points that we embed
pts = rand(3, 100)
E = embed(pts)

# How many neighbors to consider?
k = 3

# Which variables go into which marginals?
target_future = [1]
target_presentpast = [2]
source_presentpast = [3]
conditioned_presentpast = Int[]
v = TEVars(target_future,
            target_presentpast,
            source_presentpast,
            conditioned_presentpast)


# On raw points, default metric
@test transferentropy_kraskov(pts, k, v) |> isfinite
@test transferentropy_kraskov(pts, k, target_future,
                                    target_presentpast,
                                    source_presentpast,
                                    conditioned_presentpast) |> isfinite

# On embeddings, default metric
@test transferentropy_kraskov(E, k, v) |> isfinite
@test transferentropy_kraskov(E, k, target_future,
                                    target_presentpast,
                                    source_presentpast,
                                    conditioned_presentpast) |> isfinite

# Specifying the distance metric
@test transferentropy_kraskov(pts, k, v; metric = Chebyshev()) |> isfinite
@test transferentropy_kraskov(pts, k, v; metric = Chebyshev()) |> isfinite
@test transferentropy_kraskov(E, k, v; metric = Chebyshev()) |> isfinite
@test transferentropy_kraskov(E, k, v; metric = Chebyshev()) |> isfinite
