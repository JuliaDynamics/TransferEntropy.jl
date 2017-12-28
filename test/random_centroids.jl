using Distributions
using TransferEntropy
using PlotlyJS
#addprocs(2)


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

using BenchmarkTools

@time jointtest()

@time joint =
Py, Pxy, Pxz, = TransferEntropy.marginaldists(unique(bins[find(measure .> 0), :], 1), measure)
@test size(joint) == size(Py) == size(Pxy) == size(Pxz)

dist = Normal()

function te_over_binsizes(binsizes; npts::Int = 1000)
    dist = Normal()
    centroids = rand(dist, npts, 3)
    invmeasure = abs.(rand(dist, npts))
    invmeasure[randperm(400)[50]] = 0
    invmeasure = invmeasure ./ sum(invmeasure) # normalise to true probability dist.

    TE = zeros(length(binsizes))

    count = 0
    for binsize in binsizes
        count +=1
        TE[count] = te_from_triangulation(centroids, invmeasure, binsize)
    end

    return TE
end

using Base.Threads
@show nprocs()

binsizes = 1:5
TE = te_over_binsizes(binsizes)
plot(scatter(;x = binsizes, y = TE))


#@show te
#binsizes = 1:100
#TE = zeros(size(binsizes))
#@testset "Transfer entropy over bin sizes" begin
#    @testset "k for $k" for k in binsizes
#        @show k
#        te = TransferEntropy.te_from_triangulation(centroids, invmeasure, k)
#3        @test te >= 0
#        TE[k] = te
#    end
#end
#te_from_triangulation(centroids, invmeasure, 20)
