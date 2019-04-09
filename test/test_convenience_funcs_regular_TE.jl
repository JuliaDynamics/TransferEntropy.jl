using Test, TransferEntropy

x, y = rand(100), rand(100)

###########################################
# Set `dim` and infer `k`, `l` and `m`.
###########################################
tol = 1e-12

# Only with time series
@test all(transferentropy(x, y) .>= 0 - tol)
@test all(transferentropy(x, y, dim = 3) .>= 0 - tol)
@test all(transferentropy(x, y, dim = 4) .>= 0 - tol)
@test_throws ArgumentError transferentropy(x, y, dim = 2)

# Only with time series
@test transferentropy(x, y, RectangularBinning(10)) >= 0 - tol
@test all(transferentropy(x, y, [RectangularBinning(x) for x in 2:4]) .>= 0 - tol)
@test all(transferentropy(x, y, [RectangularBinning(x) for x in 2:4],  dim = 3) .>= 0 - tol)
@test all(transferentropy(x, y, [RectangularBinning(x) for x in 2:4], dim = 4) .>= 0 - tol)
@test_throws ArgumentError transferentropy(x, y, [RectangularBinning(x) for x in 2:4], dim = 2)


###########################################
# Infer `dim` from `k`, `l` and `m`.
###########################################

# Only with time series
@test all(transferentropy(x, y, 1, 1, 1) .>= 0 - tol)
@test all(transferentropy(x, y, 1, 2, 1) .>= 0 - tol)
@test_throws ArgumentError transferentropy(x, y, 1, 1, 0)

# Only with time series
@test transferentropy(x, y, RectangularBinning(10), 1, 1, 1) >= 0 - tol
@test all(transferentropy(x, y, [RectangularBinning(x) for x in 2:4], 1, 1, 1) .>= 0 - tol)
@test all(transferentropy(x, y, [RectangularBinning(x) for x in 2:4], 1, 2, 1) .>= 0 - tol)

@test_throws ArgumentError transferentropy(x, y, [RectangularBinning(x) for x in 2:4], 1, 0, 1)
