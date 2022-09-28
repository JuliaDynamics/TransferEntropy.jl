using DelayEmbeddings
import Entropies: genentropy

function norm_minmax(v; ϵ = 0.001)
    mini, maxi = minimum(v), maximum(v)

    if mini == maxi
        l = length(v)
        return [1/l for i in v]
    else
        ϵ .+ (1-2ϵ)*(v .- mini) ./  (maxi - mini)
    end
end

"""
    symbolize_by_amplitude(x::AbstractVector; A = 0.5, m::Int = length(a)) → Λ

Symbolize the state vector `x` to according to the amplitudes of its elements:

```math
\\Lambda_i = \\dfrac{A}{m} \\sum_{k=1}^m |x_k| + \\dfrac{1-A}{m-1} \\sum_{k=2}^m|x_{k} - x_{k-1}|,
```

where ``A = 0.0`` emphasizes amplitude changes, ``A = 1.0`` emphasizes only average
values, and ``A = 0.5`` equally emphasizes average values and amplitude changes.
"""
function symbolize_by_amplitude end

function symbolize_by_amplitude(x::AbstractVector; A = 0.5, m::Int = length(x))
    f = (A / m) * sum(abs.(x)) + (1 - A)/(m - 1) * sum(abs.(diff(x)))
end

"""

    AmplitudeSymbolization(; n::Int = 2)

A symbolization scheme where the state vectors `xᵢ ∈ x` of a multivariate time series `x`
first are transformed using [`symbolize_by_amplitude`](@ref). Then, the transformed `xᵢ`
are mapped to the unit interval, which is then divided into `n` equal-length bins. Finally,
each `xᵢ` is assined an integer based on which bin `xᵢ` falls in.

A variant of this approach was introduced in Azami & Escudero (2016) to compute
amplitude-aware permutation entropy (see [`SymbolicAmplitudeAwarePermutation`](@ref)),
in the context of probability estimation - not as a symbolization scheme in itself.

[^Azami2016]: Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy: Illustration in spike detection and signal segmentation. Computer methods and programs in biomedicine, 128, 40-51.

See also: [`Entropies.symbolize`](@ref).
"""
Base.@kwdef struct AmplitudeSymbolization{I <: Integer, T}
    n::I = 2
    A::T = 0.5
end

"""
# Amplitude-sensitive symbolization

    symbolize(x::AbstractDataset, s::AmplitudeSymbolization(; n::Int = 2, A = 0.5)) → Vector{Int}

Symbolize each `xᵢ ∈ x` using the amplitude-sensitive symbolization scheme `s`, which uses
a modified version of the amplitude-codification in Azami & Escudero (2016)[^Azami2016].

## Examples

```jldoctest; setup = :(using Entropies)
using DelayEmbeddings

x1 = [4, 9, 6, 1, 2];
x2 = [10, 4, 6, 10, 6];
x = Dataset(x1, x2);
symbolize(x, AmplitudeSymbolization(n = 2))

5-element Vector{Int64}:
 2
 2
 1
 2
 1
```

[^Azami2016]: Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy: Illustration in spike detection and signal segmentation. Computer methods and programs in biomedicine, 128, 40-51.

See also: [`AmplitudeSymbolization`](@ref).
"""
function symbolize(
        x::AbstractDataset,
        s::AmplitudeSymbolization = AmplitudeSymbolization(; n = 2, A = 0.5)
        )

    # Symbolize the state vectors `xᵢ ∈ x` by their amplitude information.
    Λ = symbolize_by_amplitude.(x.data, A = s.A)

    # We want the amplitude information to be encoded as integers, so normalize such that
    # Λᵢ ∈ [0, 1] ∀ Λᵢ ∈ Λ.
    Λnorm = norm_minmax(Λ)

    # Divide the unit interval into `N` equal-size bins, assign an integer to each bin,
    # and then map each Λᵢ to one of those bins (integers).
    return ceil.(Int, Λnorm ./ (1 / s.n))
end

"""
    AmplitudeAndPermutation(; n::Int = 2, m::Int = 3, τ::Int = 1, A::Real = 0.5,
        lt = Entropies.isless_rand)

A symbolic probability estimator that combines permutation information (Bandt & Pompe, 2002)
with amplitude information. `A` decides the relative weighting of amplitude values within
state vectors, where where `A = 0.0` emphasizes amplitude changes, `A = 1.0` emphasizes
only average values, and `A = 0.5`` equally emphasizes average values and amplitude
changes.

`AmplitudeAndPermutation` can be used both for symbolization and for estimating entropy.
The estimator is inspired by the amplitude-aware entropy estimator by
Azami & Escudero (2016)[^Azami2016] ([`SymbolicAmplitudeAwarePermutation](@ref)), but
instead of only considering amplitudes during probability computations, this estimator
explicitly symbolizes state vector amplitude information.

The motif length `m` must be ≥ 2. By default `m = 2`, which is the shortest
possible permutation length which retains any meaningful dynamical information.

See also: [`symbolize`](@ref), [`symbolize_by_amplitude`](@ref).

[^BandtPompe2002]: Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural complexity measure for time series." Physical review letters 88.17 (2002): 174102.
[^Azami2016]: Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy: Illustration in spike detection and signal segmentation. Computer methods and programs in biomedicine, 128, 40-51.
"""
struct AmplitudeAndPermutation{T} <: Entropies.PermutationProbabilityEstimator
    n::Int
    m::Int
    τ::Int
    A::T
    lt::Function

    function AmplitudeAndPermutation(;n::Int = 2, m = 3, τ = 1, A::T = 0.5, lt = Entropies.isless_rand) where T
        n >= 2 || throw(ArgumentError("The number of amplitude intervals N must be at least 2 (got n=$(n)), otherwise no amplitude information is distinguished."))
        0.0 <= A <= 1.0 || throw(ArgumentError("0.0 <= A <= 1.0 is required. Got A=$A."))
        new{T}(n, m, τ, A, lt)
    end
end

"""
    symbolize(x::AbstractDataset{m, T}, est::AmplitudeAndPermutation) → SVector{2, Int}

Symbolize the state vectors `xᵢ ∈ x` using both ordering information and relative
relative amplitude information of the elements of the `xᵢ`.

Returns a vector of two-element symbols `sᵢ ∈ s`, where `s[i][1]` is the permutation
symbol, and `s[i][2]` is the amplitude symbol.
"""
function symbolize(x::AbstractDataset{m, T}, est::AmplitudeAndPermutation) where {m, T}
    permsymbols = Entropies.symbolize(x, SymbolicPermutation(m = est.m, τ = est.τ, lt = est.lt))
    amplsymbols = symbolize(x, AmplitudeSymbolization(n = est.n, A = est.A))
    return [SVector{2, Int}(x, y) for (x, y) in zip(permsymbols, amplsymbols)]
end

function genentropy(x::AbstractDataset{m, T}, est::AmplitudeAndPermutation, q::U = 1;
        base = 2) where {m, T, U}

    s = symbolize(x, est)
    𝐩 = probabilities(s)
    return Entropies.genentropy(𝐩, q = q, base = base)
end

using DelayEmbeddings
using Entropies
x.data |> typeof |> supertype |> supertype
x = Dataset(rand(1000, 5))
ss = symbolize(x, AmplitudeAndPermutation())
genentropy(x, AmplitudeAndPermutation(), base = MathConstants.e)
Entropies.genentropy(x, Entropies.SymbolicAmplitudeAwarePermutation(), base = MathConstants.e)
