using Distributions
using TransferEntropy

function jointtest()
    # Test three-dimensional case
    npts = 500
    dim = 3

    dist = Normal()
    centroids = rand(dist, npts , 3)
    invmeasure = abs.(rand(dist, npts))
    invmeasure[randperm(500)[1:50]] = 0
    invmeasure = invmeasure ./ sum(invmeasure) # normalise to true probability dist.

    n = 1000
    bins, measure = get_nonempty_bins(centroids, invmeasure, [n, n, n])
    TransferEntropy.jointdist(bins, measure)
end


jointtest()
@time jointtest()
