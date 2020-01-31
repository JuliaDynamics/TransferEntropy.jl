x, y, z = rand(50), rand(50), rand(50)

est_vf = VisitationFrequency()
est_to = TransferOperatorGrid()
est_nn = NearestNeighbourMI()

p = EmbeddingTE()

# Regular transfer entropy
@test transferentropy(x, y, p, est_vf)
@test transferentropy(x, y, z, p, est_vf)

# Conditional transfer entropy
