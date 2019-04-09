using Test, TransferEntropy

x, y, z = rand(100), rand(100), rand(100)

###########################################
# Set `dim` and infer `k`, `l` and `m`.
###########################################
tol =  1e-12
# Only with time series
@test all(transferentropy(x, y, z) .>= 0)
@test all(transferentropy(x, y, z, dim = 5) .>= 0 - tol)
@test_throws ArgumentError transferentropy(x, y, z, dim = 3)

# Only with time series
@test transferentropy(x, y, z, RectangularBinning(10)) >=  0 - tol
@test all(transferentropy(x, y, z, [RectangularBinning(x) for x in 2:4]) .>=  0 - tol)
@test all(transferentropy(x, y, z, [RectangularBinning(x) for x in 2:4], dim = 4) .>=  0 - tol)
@test all(transferentropy(x, y, z, [RectangularBinning(x) for x in 2:4], dim = 5) .>=  0 - tol)

@test_throws ArgumentError transferentropy(x, y, z, [RectangularBinning(x) for x in 2:4], dim = 2)


###########################################
# Infer `dim` from `k`, `l` and `m`.
###########################################

# Only with time series
@test all(transferentropy(x, y, z, 1, 1, 1, 1) .>= 0 - tol)
@test all(transferentropy(x, y, z, 1, 2, 1, 1) .>= 0 - tol)
@test_throws ArgumentError transferentropy(x, y, z, 1, 1, 0, 1)

# Only with time series
@test transferentropy(x, y, z, RectangularBinning(10), 1, 1, 1, 1) >= 0 - tol
@test all(transferentropy(x, y, z, [RectangularBinning(x) for x in 2:4], 1, 1, 1, 1) .>= 0 - tol)
@test all(transferentropy(x, y, z, [RectangularBinning(x) for x in 2:4], 1, 2, 1, 1) .>= 0 - tol)

@test_throws ArgumentError transferentropy(x, y, z, [RectangularBinning(x) for x in 2:4], 1, 0, 1, 1)
