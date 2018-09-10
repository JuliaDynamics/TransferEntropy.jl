using Interpolations, Cubature

"""
    ∫(x, y, a, b)

Represent `y` as a function of `x` using a linear
interpolation, then take the integral from `a` to `b`.
"""
function ∫(x, y, a, b)
    lininterp = LinearInterpolation(x, y)
    f(a) = lininterp(a)
    hquadrature(f, a, b)
end

"""
    te_integral(binsizes, te_estimates)

Compute scaled the area under the curve for transfer entropy
estimates across bin sizes.
"""
function te_integral(binsizes, te_estimates)
    bmin = minimum(binsizes)
    bmax = maximum(binsizes)
    Δb = bmax - bmin
    sortinds = sortperm(binsizes)
    binsizes = binsizes[sortinds]
    te_est_sorted = te_estimates[sortinds]

    return (1/Δb) * ∫(binsizes, te_est_sorted, bmin, bmax)[1]
end

"""
    ∫te(binsizes, te_estimates)

Compute scaled the area under the curve for transfer entropy
estimates across bin sizes.
"""
∫te(binsizes, te_estimates) = te_integral(binsizes, te_estimates)
