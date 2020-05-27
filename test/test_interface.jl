x, y, z = rand(50), rand(50), rand(50)

est_vf = VisitationFrequency()
est_to = TransferOperatorGrid()
est_nn = NearestNeighborMI()

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
@test transferentropy(x, y, E, est_to) isa Real
@test transferentropy(x, y, z, E, est_to) isa Real
