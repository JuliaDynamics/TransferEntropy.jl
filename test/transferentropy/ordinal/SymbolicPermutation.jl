# This estimator has specialized methods, so gets its own test file.
@testset "SymbolicPermutation" begin
    m, τ = 3, 1
    est = SymbolicPermutation(m = m, τ = τ)
    e = Shannon(base = 2)
    te =  transferentropy(e, est, s, t)
    tec = transferentropy(e, est, s, t, c)
    @test te isa Real
    @test tec isa Real

    πS, πT, πC = ([zeros(Int, length(s)-(m-1)*τ) for x in 1:3]...,)
    te_inplace =  transferentropy!(πS, πT, e, est, s, t)
    tec_inplace = transferentropy!(πS, πT, πC, e, est, s, t, c)
    @test te_inplace isa Real
    @test tec_inplace isa Real
end
