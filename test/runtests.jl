#if lowercase(get(ENV, "CI", "false")) == "true"
#    include("install_dependencies.jl")
#end

#using CausalityToolsBase
using Test, TransferEntropy, Entropies
KDTree = Entropies.KDTree
BruteForce = Entropies.BruteForce

@testset "Mutual information" begin
    x = rand(100)
    y = rand(100)
    z = Dataset(rand(100, 2))
    w = Dataset(rand(100, 3))

    # Estimators for which Renyi entropies can be used
    est = VisitationFrequency(RectangularBinning(0.2))
    @test mutualinfo(x, y, est) isa Real
    @test mutualinfo(x, y, est, q = 2) isa Real
    @test mutualinfo(x, z, est) isa Real
    @test mutualinfo(z, x, est) isa Real
    @test mutualinfo(z, w, est) isa Real

    # Estimators for which Renyi entropies cannot be used
    est_kl = KozachenkoLeonenko()
    est_k = Kraskov()
    est_k1 = Kraskov1(2)
    est_k2 = Kraskov2(2)

    @test mutualinfo(x, y, est_kl) isa Real
    @test mutualinfo(x, y, est_kl, base = 2) isa Real
    @test mutualinfo(x, z, est_kl) isa Real
    @test mutualinfo(z, x, est_kl) isa Real
    @test mutualinfo(z, w, est_kl) isa Real

    @test mutualinfo(x, y, est_k) isa Real
    @test mutualinfo(x, y, est_k, base = 2) isa Real
    @test mutualinfo(x, z, est_k) isa Real
    @test mutualinfo(z, x, est_k) isa Real
    @test mutualinfo(z, w, est_k) isa Real

    @test mutualinfo(x, y, est_k1) isa Real
    @test mutualinfo(x, y, est_k1, base = 2) isa Real
    @test mutualinfo(x, z, est_k1) isa Real
    @test mutualinfo(z, x, est_k1) isa Real
    @test mutualinfo(z, w, est_k1) isa Real

    @test mutualinfo(x, y, est_k2) isa Real
    @test mutualinfo(x, y, est_k2, base = 2) isa Real
    @test mutualinfo(x, z, est_k2) isa Real
    @test mutualinfo(z, x, est_k2) isa Real
    @test mutualinfo(z, w, est_k2) isa Real
end

@testset "Conditional mutual information" begin
    s, t, c = rand(100), rand(100), rand(100)
    est_knn = Kraskov1(2)
    est_bin = RectangularBinning(3)
    # binning estimator yields non-negative values
    @test conditional_mutualinfo(s, t, c, est_bin, q = 2) isa Real
    @test conditional_mutualinfo(s, t, c, est_bin, q = 2) >= 0.0
    # verify formula I(X, Y | Z) = I(X; Y, Z) - I(X, Z)
    @test conditional_mutualinfo(s, t, c, est_bin, base = 2) ≈ 
    mutualinfo(s, Dataset(t, c), est_bin, base = 2) - mutualinfo(s, c, est_bin, base = 2)

    @test conditional_mutualinfo(s, t, c, est_knn) isa Real
    @test conditional_mutualinfo(s, t, c, est_knn, base = 2) ≈ 
        mutualinfo(s, Dataset(t, c), est_knn, base = 2) - mutualinfo(s, c, est_knn, base = 2)
   
    # Different types of input
    @test conditional_mutualinfo(s, Dataset(t, c), c, est_bin) isa Real
    @test conditional_mutualinfo(Dataset(s, t), Dataset(t, c), c, est_bin) isa Real
    @test conditional_mutualinfo(Dataset(s, t), Dataset(t, c), Dataset(c, s), est_bin) isa Real
    @test conditional_mutualinfo(s, Dataset(t, c), Dataset(c, s), est_bin) isa Real
    @test conditional_mutualinfo(s, t, Dataset(c, s), est_bin) isa Real
    @test conditional_mutualinfo(Dataset(s, t), t, c, est_bin) isa Real
end

@testset "Transfer entropy" begin 
    s, t, c = rand(100), rand(100), rand(100)

    println("Starting transfer entropy tests...")
    @testset "Generalized Renyi transfer entropy" begin
        # Straight-forward estimators
        est_vf = VisitationFrequency(RectangularBinning(4))
        ests = [
            est_vf,
            TransferOperator(RectangularBinning(5)),
            Hilbert(source = Phase(), target = Amplitude(), est_vf),
            Hilbert(source = Amplitude(), target = Amplitude(), est_vf),
            Hilbert(source = Phase(), target = Phase(), est_vf),
            Hilbert(source = Phase(), target = Phase(), cond = Amplitude(), est_vf),
            NaiveKernel(0.5, KDTree),
            NaiveKernel(0.5, BruteForce)
        ]

        @testset "Generalized Renyi transfer entropy $(ests[i])"  for i in 1:length(ests)
            println(ests[i])
            est = ests[i]
            te =  transferentropy(s, t, est, q = 2, base = 2)
            tec = transferentropy(s, t, c, est, q = 1, base = 3)

            @test te isa Real
            @test tec isa Real

            if est isa VisitationFrequency
                @test te > 0 || te ≈ 0
                @test tec > 0 || tec ≈ 0
            end
        end

        @testset "SymbolicPermutation" begin
            m, τ = 3, 1
            est = SymbolicPermutation(m = m, τ = τ)
            te =  transferentropy(s, t, est, q = 2, base = 2)
            tec = transferentropy(s, t, c, est, q = 1, base = 3)

            S, T, C = ([zeros(Int, length(s)-(m-1)*τ) for x in 1:3]...,)
            te_inplace =  transferentropy!(S, T, s, t, est, q = 2, base = 2)
            tec_inplace = transferentropy!(S, T, C, s, t, c, est, q = 1, base = 3)

            @test te isa Real
            @test tec isa Real
            @test te_inplace isa Real
            @test tec_inplace isa Real
        end
    end


    
    @testset "Shannon transfer entropy" begin 
        ests = [
            Kraskov(),
            KozachenkoLeonenko()
        ]
        @testset "Nearest neighbors $(ests[i])"  for i in 1:length(ests)
            println(ests[i])
            est = ests[i]
            te =  transferentropy(s, t, est, base = 2)
            tec = transferentropy(s, t, c, est, base = 3)
            @test te isa Real
            @test tec isa Real

        end
    end

    @testset "Automated estimators" begin
        @testset "Variable exclusion" begin
             # Use periodic signals, so we also can test variable selection methods,
            # which for sensible testing, need to have their autocorrelation minima 
            # > 1.
            s = sin.(1:100) .+ rand(100)
            t = sin.(1:100) .+ rand(100)
            c = sin.(1:100) .+ rand(100)

            τexclude = 1
            vars = TransferEntropy.construct_candidate_variables([s], [t], τexclude = nothing)
            vars_ex = TransferEntropy.construct_candidate_variables([s], [t], τexclude = τexclude)
            fvars = Iterators.flatten(vars[1][1:end-1]) |> collect
            fvars_ex = Iterators.flatten(vars_ex[1][1:end-1]) |> collect
            @test τexclude ∈ abs.(fvars)
            @test τexclude ∉ abs.(fvars_ex)
            @test length(fvars) > length(fvars_ex)

            vars = TransferEntropy.construct_candidate_variables([s], [t], [c], τexclude = nothing)
            vars_ex = TransferEntropy.construct_candidate_variables([s], [t], [c], τexclude = τexclude)
            fvars = Iterators.flatten(vars[1][1:end-1]) |> collect
            fvars_ex = Iterators.flatten(vars_ex[1][1:end-1]) |> collect
            @test τexclude ∈ abs.(fvars)
            @test τexclude ∉ abs.(fvars_ex)
            @test length(fvars) > length(fvars_ex)
        end

        est = VisitationFrequency(RectangularBinning(3))
        te_st, params_st = bbnue(s, t, est, include_instantaneous = true)
        te_stc, params_stc = bbnue(s, t, c, est, include_instantaneous = false)
        @test te_stc isa Float64
        @test te_stc >= 0.0
    end

    

end
