export VisitationFrequency, _transferentropy

import StatsBase
import CausalityToolsBase: joint_visits, marginal_visits, non0hist


"""
    VisitationFrequency(; b = 2, 
        summary_statistic = StatsBase.mean, 
        binning = ExtendedPalusLimit())

An transfer entropy estimator which computes transfer entropy over a 
dicretization of an appropriate delay reconstruction from the 
input time series [^Diego2019]. The invariant probabilities over the partition are estimated 
using a simple counting approach.

## Fields 

- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).
- **`summary_statistic::Function = StatsBase.mean`**: The summary statistic to use if multiple discretization schemes are given.
- **`binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic}`**: 
    The discretization scheme. Can either be fixed (i.e. one or more `RectangularBinning` instances),
    or a `BinningHeuristic`. In the latter case, the binning is determined from the input data.

## Examples

```julia
# Transfer entropy in bits (logarithm to base 2), and partition 
estimator = VisitationFrequency(b = 2, binning = RectangularBinning(5)) 
```

[^Diego2019]: Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
Base.@kwdef struct VisitationFrequency <: BinningTransferEntropyEstimator
    """ The base of the logarithm usen when computing transfer entropy. """
    b::Number = 2.0

    """ The summary statistic to use if multiple discretization schemes are given """
    summary_statistic::Function = StatsBase.mean

    """ The discretization scheme. """
    binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic} = ExtendedPalusLimit()

    VisitationFrequency(b, summary_statistic, binning) = new(b, summary_statistic, binning)
end

import .._transferentropy
import ..TEVars

function _transferentropy(pts, vars::TEVars, binning::RectangularBinning, estimator::VisitationFrequency)

    # Collect variables for the marginals 
    C = vars.C
    XY = [vars.𝒯; vars.T; C]
    YZ = [vars.T; vars.S; C]
    Y =  [vars.T; C]
    
    # Find the bins visited by the joint system (and then get 
    # the marginal visits from that, so we don't have to encode 
    # bins multiple times). 
    joint_bin_visits = joint_visits(pts, binning)

    # Compute visitation frequencies for the nonempty bins.
    p_Y = non0hist(marginal_visits(joint_bin_visits, Y))
    p_XY = non0hist(marginal_visits(joint_bin_visits, XY))
    p_YZ = non0hist(marginal_visits(joint_bin_visits, YZ))
    p_joint = non0hist(joint_bin_visits)
    
    # Base of the logarithm
    b = estimator.b

    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(p_joint, b)
end