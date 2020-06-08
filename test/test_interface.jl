x, y, z = rand(50), rand(50), rand(50)

est_vf = VisitationFrequency()
est_to = TransferOperatorGrid()
est_nn = NearestNeighborMI()
est_perm = SymbolicPerm()
est_permaa = SymbolicAmplitudeAware()

est_simplex_point = SimplexEstimator(SimplexPoint(), VisitationFrequency())
est_simplex_exact = SimplexEstimator(SimplexExact(), TransferOperatorGrid(binning = RectangularBinning(3)))

x, y = rand(25), rand(25)
E = EmbeddingTE()

# VisitationFrequency
# -------------------
@test transferentropy(x, y, E, est_vf) isa Real
@test transferentropy(x, y, z, E, est_vf) isa Real

# TransferOperatorGrid
# -------------------
@test transferentropy(x, y, E, est_to) isa Real
@test transferentropy(x, y, z, E, est_to) isa Real

# NearestNeighborMI
# -------------------
@test transferentropy(x, y, E, est_nn) isa Real
@test transferentropy(x, y, z, E, est_nn) isa Real

# SymbolicPerm
# -------------------
@test transferentropy(x, y, E, est_perm) isa Real
@test transferentropy(x, y, z, E, est_perm) isa Real

# SymbolicAmplitudeAware
# -------------------
@test transferentropy(x, y, E, est_permaa) isa Real
@test transferentropy(x, y, z, E, est_permaa) isa Real

# SimplexEstimator 
# ----------------
@test transferentropy(x, y, E, est_simplex_exact) isa Real
@test transferentropy(x, y, z, E, est_simplex_exact) isa Real
@test transferentropy(x, y, E, est_simplex_point) isa Real
@test transferentropy(x, y, z, E, est_simplex_point) isa Real