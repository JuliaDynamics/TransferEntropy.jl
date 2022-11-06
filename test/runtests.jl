using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
@testset "TransferEntropy.jl" begin
    # Probability estimators
    testfile("mutualinfo/mutualinfo.jl")
    testfile("conditional_mutualinfo/conditional_mutualinfo.jl")

end
