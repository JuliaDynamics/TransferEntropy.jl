# Kraskov estimator

The Kraskov estimator decomposes the transfer entropy into a sum of mutual
informations, which are estimated using the I2 algorithm from Kraskov et al. [1].

```@setup kraskov
using StateSpaceReconstruction
using TransferEntropy
```

## From an array of points (each column being a point)
```@example kraskov
pts = rand(3, 100)
k = 3

target_future = [1]
target_presentpast = [2]
source_presentpast = [3]
conditioned_presentpast = Int[]

v = TEVars(target_future,
            target_presentpast,
            source_presentpast,
            conditioned_presentpast)

```

## References
1. Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
mutual information." Physical review E 69.6 (2004): 066138.
