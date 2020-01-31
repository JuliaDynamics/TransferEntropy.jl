using DelayEmbeddings
using TransferEntropy

# Estimate transfer entropy from scratch from a random
# set of points `n_realizations` times. If everything
# works as expected, there should be no negative transfer
# entropy values.
ts_length = 100

@testset "3D #$i" for i in 1:n_realizations
	x = rand(100)
	y = rand(100)
	D = Dataset(x, y)
	pts = customembed(D, Positions(2, 2, 1), Lags(1, 0, 0))
	v = TEVars(𝒯 = [1], T = [2], S = [3])
	binning_scheme = RectangularBinning([2, 4, 5])

	@test transferentropy_visitfreq(pts, binning_scheme, v, b = 2) >= 0
end

@testset "4D #$i" for i in 1:n_realizations
	x = rand(100)
	y = rand(100)
	D = Dataset(x, y)
	pts = customembed(D, Positions(2, 2, 2, 1), Lags(1, 0, -2, 0))
	v = TEVars(𝒯 = [1], T = [2, 3], S = [4])
	binning_scheme = RectangularBinning([2, 4, 5, 5])
	
	@test transferentropy_visitfreq(pts, binning_scheme, v, b = 2) >= 0
end


@testset "5D #$i" for i in 1:n_realizations
	x = rand(100)
	y = rand(100)
	D = Dataset(x, y)
	pts = customembed(D, Positions(2, 2, 2, 2, 1), Lags(1, 0, -2, -5, 0))
	v = TEVars(𝒯 = [1], T = [2, 3], S = [4, 5])
	binning_scheme = RectangularBinning([2, 4, 5, 5, 3])

	@test transferentropy_visitfreq(pts, binning_scheme, v, b = 2) >= 0
end