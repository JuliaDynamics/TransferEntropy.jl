export SymbolicAmplitudeAware

import DelayEmbeddings: Dataset
import Combinatorics: nthperm

"""
    SymbolicAmplitudeAware(b = 2, m::Int = 3, n::Int = 5, A::Real = 0.5) <: SymbolicTransferEntropyEstimator

A symbolic transfer entropy estimator that uses an amplitude-aware 
symbolization of the delay reconstruction vectors. 

Where [`SymbolicPerm`](@ref) only considers up-down patterns of the data during
the symbolization step (disregarding amplitude information), this 
estimator subdivides each symbol motif (permutation pattern) into `n` further 
sub-symbols according to the amplitudes of the delay vector components, as well 
as the differences in consecutive delay vector terms. Thus, whereas there 
are ``m!`` different motis for [`SymbolicPerm`](@ref), for [`SymbolicAmplitudeAware`](@ref)
there are ``m!n`` possible motifs for each delay reconstruction marginal.

The parameter `A ∈ [0, 1]` controls whether only the delay vector component 
amplitudes are considered (`A = 1`), or if the differences between consecutive 
terms are emphasized (`A = 0`). If `A ∈ (0, 1)`, then both magnitudes and 
differences are considered during symbolization, with `A = 0.5` equally balancing 
amplitudes and differences in amplitudes. 

Except for this difference in the initia symbolization step, the transfer entropy 
estimation procedure is the same as for [`SymbolicPerm`](@ref).
"""
struct SymbolicAmplitudeAware <: SymbolicTransferEntropyEstimator
    b::Real
    m::Int
    n::Int # the number of submotifs after quantizing
    A::Real # the weighting factor.
    
    function SymbolicAmplitudeAware(; b::Real = 2, m::Int = 3, n::Int = 5, A::Real = 0.5)
        m >= 2 || throw(ArgumentError("Dimensions of individual marginals must be at least 2. Otherwise, symbol sequences cannot be assigned to the marginals. Got m=$(m)."))
        n >= 1 || throw(ArgumentError("The number of bins must be >= 1."))
        1 >= A >= 0 || throw(ArgumentError("Need 1 >= A >= 0. Got $(A)."))
        new(b, m, n, A)
    end
end

function norm_minmax(v, ϵ = 0.001)
    mini, maxi = minimum(v), maximum(v)
    
    
    if mini == maxi
        l = length(v)
        return [1/l for i in v]
    else
        ϵ .+ (1-2ϵ)*(v .- mini) ./  (maxi - mini)
    end
end

function quantize(v; n_submotifs = 5)
    mini, maxi = minimum(v), maximum(v)
    r = maxi - mini
    Δx = 1/n_submotifs
    
    x = zero(v)
    for i = 1:length(x)
        x[i] = Δx*floor(Int, v[i]/Δx)
    end
    
    return x
end

"""
A = 1 emphasizes only average values
A = 0 emphasizes changes in amplitude values
A = 0.5 emphasizes equally average values and changes in the amplitude values
"""
function AAPE_norm(a, A=0.5, d::Int = length(a))
    f = (A/d)*sum(a) + (1-A)/(d-1)*sum(diff(a))
end

""" 
Amplitude aware symbolization of the points of `E`.
"""
function symbolize(E::Dataset, method::SymbolicAmplitudeAware)
    m = length(E[1])
    n = length(E)

    # Pre-allocate symbol vector
    symb = zeros(n)
    
    #= 
    Loop over embedding vectors `E[i]`, find the indices `p` that sort them
    and get the corresponding integer `k` that generated the 
    permutation `p`. That integer is the symbol for the embedding vector
    `E[i]`.
    =#
    sp = zeros(Int, m)

    # Compute normalization, map to range [0, 1] and 
    # quantize/bin into method.n different values
    f = AAPE_norm.(E.data, method.A)
    nf = norm_minmax(f)
    submotifs = quantize(nf, n_submotifs = method.n)
    
    @inbounds for i = 1:n
        # The permutation part of the symbol
        sortperm!(sp, E[i])
        symb[i] = encode_pattern(sp, m) + submotifs[i]
    end
    
    return symb
end