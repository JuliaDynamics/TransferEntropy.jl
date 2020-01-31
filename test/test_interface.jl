x, y, z = rand(50), rand(50), rand(50)

est_vf = VisitationFrequency()
est_to = TransferOperatorGrid()
est_nn = NearestNeighbourMI()

p = EmbeddingTE()

# VisitationFrequency
# -------------------
@test transferentropy(x, y, p, est_vf) isa Real
@test transferentropy(x, y, z, p, est_vf) isa Real

# TransferOperatorGrid
# -------------------
@test transferentropy(x, y, p, est_to) isa Real
@test transferentropy(x, y, z, p, est_to) isa Real

# Conditional transfer entropy
