import PerronFrobenius: AbstractTriangulationInvariantMeasure
import CausalityToolsBase: RectangularBinning, CustomReconstruction
import StateSpaceReconstruction: Simplex, generate_interior_points
import StaticArrays: SVector
import Distances: Metric, Chebyshev
import StatsBase 

export transferentropy, 
    BinningTransferEntropyEstimator, 
    TransferEntropyEstimator, 
        TransferOperatorGrid, 
        VisitationFrequency, 
        NearestNeighbourMI

"""
    TransferEntropyEstimator

An abstract type for transfer entropy estimators. This type has several concrete subtypes
that are accepted as inputs to the [`transferentropy`](@ref) methods. 

- [`VisitationFrequency`](@ref)
- [`TransferOperatorGrid`](@ref)
- [`NearestNeighbourMI`](@ref)
"""
abstract type TransferEntropyEstimator end 

function Base.show(io::IO, estimator::TransferEntropyEstimator)
    s = "$(typeof(estimator))($(estimator.b))"
    print(io, s)
end

"""
    BinningTransferEntropyEstimator <: TransferEntropyEstimator

An abstract type for transfer entropy estimators that works on a discretization 
of the [reconstructed state space](@ref custom_delay_reconstruction). Has the following concrete subtypes

- [`VisitationFrequency`](@ref)
- [`TransferOperatorGrid`](@ref)

## Used by

Concrete subtypes are accepted as inputs by

- [`transferentropy`](@ref te_estimator_rectangular) (low-level method)
"""
abstract type BinningTransferEntropyEstimator end 


"""
    NearestNeighbourMI(k1::Int = 2, k2::Int = 3, metric::Metric = Chebyshev, b::Number)

A transfer entropy estimator using counting of nearest neighbours nearest neighbours 
to estimate mutual information over an appropriate 
[custom delay reconstruction](@ref custom_delay_reconstruction) of the input data. 
(the method from Kraskov et al. (2004)[^1], as implemented in Diego et al. (2019)[^2]).

## Fields 

- **`k1::Int = 2`**: The number of nearest neighbours for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- **`k2::Int = 3`**: The number of nearest neighbours for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- **`metric::Metric = Chebyshev()`**: The metric used for distance computations.
- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).

## References

[^1]:
    Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
    mutual information." Physical review E 69.6 (2004): 066138.
[^2]:
    Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation 
    using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
@Base.kwdef struct NearestNeighbourMI <: TransferEntropyEstimator
    k1::Int = 2
    k2::Int = 3
    metric::Metric = Chebyshev()
    b::Number = 2
end

function Base.show(io::IO, estimator::NearestNeighbourMI)
    k1 = estimator.k1
    k2 = estimator.k2
    metric = estimator.metric
    b = estimator.b
    s = "$(typeof(estimator))(b=$(b), k1=$(k1), k2=$(k2), metric=$(typeof(metric)))"
    print(io, s)
end

"""
    TransferOperatorGrid(; b::Number = 2, summary_statistic = StatsBase.mean, 
        binning = ExtendedPalusLimit())

An transfer entropy estimator which computes transfer entropy over a 
dicretization of an appropriate [delay reconstruction](@ref custom_delay_reconstruction) of the input data.
Invariant probabilities over the partition are computed using an approximation to the transfer (Perron-Frobenius) 
operator over the grid [1], which explicitly gives the transition probabilities between states. 
The transfer entropy is computed using the logarithm to base `b`. 

## Fields 

- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).
- **`summary_statistic::Function = StatsBase.mean`**: The summary statistic to use if multiple discretization schemes are given.
- **`binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic} = ExtendedPalusLimit()`**: 
    The discretization scheme. Can either be fixed (i.e. one or more `RectangularBinning` instances),
    or a `BinningHeuristic`. In the latter case, the binning is determined from the input data.

## References

[^1]:
    Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation 
    using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
"""
Base.@kwdef struct TransferOperatorGrid <: BinningTransferEntropyEstimator
    """ The base of the logarithm usen when computing transfer entropy. """
    b::Number = 2.0

    """ The summary statistic to use if multiple discretization schemes are given """
    summary_statistic::Function = StatsBase.mean

    """ The discretization scheme. """
    binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic} = ExtendedPalusLimit()

    TransferOperatorGrid(b, summary_statistic, binning) = new(b, summary_statistic, binning)
end

"""
    VisitationFrequency(; b::Number = 2)

An transfer entropy estimator which computes transfer entropy over a 
dicretization of an appropriate [delay reconstruction](@ref custom_delay_reconstruction) from the 
input time series [^1]. The invariant probabilities over the partition are estimated 
using a simple counting approach.

## Fields 

- **`b::Number = 2`**: The base of the logarithm, controlling the unit of the transfer 
    entropy estimate (e.g. `b = 2` will give the transfer entropy in bits).
- **`summary_statistic::Function = StatsBase.mean`**: The summary statistic to use if multiple discretization schemes are given.
- **`binning::Union{RectangularBinning, Vector{RectangularBinning}, BinningHeuristic}`**: 
    The discretization scheme. Can either be fixed (i.e. one or more `RectangularBinning` instances),
    or a `BinningHeuristic`. In the latter case, the binning is determined from the input data.

## References

[^1]:
    Diego, David, Kristian Agasøster Haaga, and Bjarte Hannisdal. "Transfer entropy computation 
    using the Perron-Frobenius operator." Physical Review E 99.4 (2019): 042212.
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

"""
    transferentropy(μ::AbstractTriangulationInvariantMeasure, vars::TEVars,
        binning_scheme::RectangularBinning; 
        estimator::BinningTransferEntropyEstimator = VisitationFrequency(), 
        n::Int = 10000) -> Float64

#### Transfer entropy using a precomputed invariant measure over a triangulated partition

Estimate transfer entropy from an invariant measure over a triangulation of 
an appropriate [generalised delay reconstruction](@ref custom_delay_reconstruction).

The invariant measure has been precomputed either as 

1. `μ = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())`, or
2. `μ = invariantmeasure(pts, TriangulationBinning(), ExactIntersection())` 

`μ` contains all the information needed to compute transfer entropy, and the estimate 
of the measure will be slightly different depending on the method. Approximate simplex 
intersections is faster, while exact simplex intersections are (very) slow. 

Note: `pts` must be a vector of states, not a vector of 
variables/(time series). Wrap your time series in a `Dataset`
first if the latter is the case.

#### Computing transfer entropy (triangulation -> rectangular partition)

Computing transfer entropy directly on a triangulation is not possible. 
To compute marginals, we need a rectangular grid. 

Therefore, first we create a cloud of points (approximately `n` points) 
with known measure by sampling points uniformly from within the simplices 
of the triangulation. Each point has measure proportional to the measure 
of the simplex from which it was sampled.  

Introducing multiple points as representatives for the partition elements 
does not introduce any bias, because in computing the 
invariant measure, we use no more information than what is encoded in the 
dynamics of the original data points. But from the invariant measure,
we can get a practically infinite amount of points to estimate transfer 
entropy from, by generating points as described above.

After sampling, a rectangular grid is superimposed on the new point cloud 
according to `binning_scheme`. 
Transfer entropy is then computed using a [`BinningTransferEntropyEstimator`].
frequency estimator on those (roughly `n`) points.

#### Common use case

The invariant measure, which encodes the dynamical information, is slow to compute over 
the triangulation, but only needs to be computed once. After that, transfer entropy may 
be estimated at multiple scales (grids) very quickly. 

This method is therefore useful if you want to explore the sensitivity of transfer entropy 
to the bin size in the final rectangular grid, when you have few observations in the time series.

### Example 

```julia
# Assume these points are an appropriate delay embedding {(x(t), y(t), y(t+1))} and 
# that we're measure transfer entropy from x -> y. 
pts = invariantize([rand(3) for i = 1:30])

v = TEVars(Tf = [3], Tpp = [2], Spp = [1])

# Compute invariant measure over a triangulation using approximate 
# simplex intersections. This is relatively slow.
μ = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())

# Compute transfer entropy from the invariant measure over multiple 
# bin sizes. This is fast, because the measure has been precomputed.
tes = map(ϵ -> transferentropy(μ, v, RectangularBinning(ϵ)), 2:50)
```
"""
function transferentropy(μ::AbstractTriangulationInvariantMeasure, vars::TEVars,
        binning_scheme::RectangularBinning; 
        estimator::BinningTransferEntropyEstimator = VisitationFrequency(b = 2), n::Int = 20000)
    
    # Get the base of the logarithm
    b = estimator.b 

    dim = length(μ.points[1])
    triang = μ.triangulation.simplexindices
    n_simplices = length(triang)
    
    simplices = [Simplex(μ.points[triang[i]]) for i = 1:n_simplices]
    
    # Find a number of points to fill each simplex with so that we 
    # obey the measure over the simplices of the triangulation.
    # The total number of points will be roughly `n`, but slightly 
    # higher because we need integer numbers of points and use `ceil`
    # for this.
    n_fillpts_persimplex = ceil.(Int, μ.measure.dist .* n)

    # Array to store the points filling the simplices
    fillpts = Vector{SVector{dim, Float64}}()
    sizehint!(fillpts, sum(n_fillpts_persimplex))

    for i = 1:n_simplices
        sᵢ = simplices[i]
        
        if n_fillpts_persimplex[i] > 0
            pts = generate_interior_points(sᵢ, n_fillpts_persimplex[i])
            append!(fillpts, [SVector{dim, Float64}(pt) for pt in pts])
        end
    end

    sizehint!(fillpts, length(fillpts))

    transferentropy(fillpts, vars, binning_scheme, estimator)
end



"""
    transferentropy(pts, vars::TEVars, ϵ::RectangularBinning, 
        estimator::BinningTransferEntropyEstimator) -> Float64

Compute the transfer entropy for a set of `pts` over the state space 
partition specified by `ϵ` (a [`RectangularBinning`](@ref) instance). 

## Fields 

- **`pts`**: An ordered set of `m`-dimensional points (`pts`) representing 
    an appropriate [generalised embedding]((@ref custom_delay_reconstruction)) 
    of some data series. Must be vector of states, not a vector of variables/time series. 
    Wrap your time series in a `DynamicalSystemsBase.Dataset` first if the latter is the case.
- **`vars::TEVars`**: A [`TEVars`](@ref) instance specifying how the `m` different 
    variables of `pts` are to be mapped into the marginals required for transfer 
    entropy computation. 
- **`ϵ::RectangularBinning`**: A [`RectangularBinning`](@ref) instance that 
    dictates how the point cloud (state space reconstruction) should be discretized. 
- **`estimator::BinningTransferEntropyEstimator`**. There are different ways of 
    computing the transfer entropy over a discretization of a point cloud.
    The `estimator` should be a valid [`TransferEntropyEstimator`](@ref)
    that works for rectangular partitions, for example `VisitationFrequency()`
    or `TransferOperatorGrid()`. The field `estimator.b` sets the base of the 
    logarithm used for the computations (e.g `VisitationFrequency(b = 2)` computes 
    the transfer entropy in bits using the [VisitationFrequency](@ref) estimator). 

## Returns 

A single number that is the transfer entropy for the discretization of `pts`
using the binning scheme `ϵ`, using the provided `estimator`. 


## Long example 

```
using TransferEntropy, DynamicalSystems
```

#### 1. Generate some example time series

Let's generate some random noise series and use those as our time series.

```julia
x = rand(100)
y = rand(100)
```

We need `pts` to be a vector of states. Therefore, collect the time series in a 
`DynamicalSystems.Dataset` instance. This way, the states of the composite system
will be represented as a `Vector{SVector}`.

```julia
raw_timeseries = Dataset(x, y)
```

#### 2. Generalised embedding

*Note: If your data are already organised in a form of a 
[generalised embedding](@ref custom_delay_reconstruction), 
where columns of the dataset correspond to lagged variables of the time series, 
you can skip to step 3.*

Say we want to compute transfer entropy from ``x`` to ``y``, and that we 
require a 4-dimensional embedding. For that, we need to decide on a 
generalised state space reconstruction of the time series. One possible choice 
is 

```math
E = \\{S_{pp}, T_{pp}, T_f \\}= \\{x_t, (y_t, y_{t-\\tau}), y_{t+\\eta} \\}
``` 

If so, we're computing the following TE

```math
TE_{x \\to y} =  \\int_E P(x_t, y_{t-\\tau} y_t, y_{t + \\eta}) \\log{\\left( \\dfrac{P(y_{t + \\eta} | (y_t, y_{t - \\tau}, x_t)}{P(y_{t + \\eta} | y_t, y_{t-\\tau})} \\right)}.
```

To create the embedding, we'll use the [`customembed`](@ref) function (check its 
documentation for a detailed explanation on how it works). This is basically just 
making lagged copies of the time series, and stacking them next to each other 
as column vectors. The order in which we arrange the lagged time series is not 
important per se, but we need to keep track of the ordering, because that 
information is crucial to the transfer entropy estimator.

According to the reconstruction we decided on above, we need to put the lagged 
time series for `y` (the target variable) in the first three columns. The lags 
for those columns are `η, 0, -τ`, in that order. Next, we need to put the 
time series for `x` (the source variable) in the fourth column (which is not 
lagged).

```julia
τ = optimal_delay(y) # find the optimal embedding lag
η = 2 # prediction lag (your choice as an analyst)
embedding_pts = customembed(raw_timeseries, Positions(2, 2, 2, 1), Lags(η, 0, -τ, 0))
```

The combination of the `Positions` instance and the `Lags` instance gives us 
the necessary information about which time series in `raw_timeseries` that 
corresponds to columns of the embedded dataset, and which lags each of the
columns have.

#### 3. Instructions to the estimator

The transfer entropy estimators needs the following about the columns of 
the generalised reconstruction of your time series (`embedding_pts` in 
our case):

- Which columns correspond to the future of the target variable (``T_f```)?
- Which columns correspond to the present and past of the target variable (``T_{pp}```)?
- Which columns correspond to the present and past of the source variable (``S_{pp}```)?
- Which columns correspond to the present/past/future of any variables 
    that we are to condition on (``C_{pp}``)?

This information is needed to ensure that marginals are properly assigned during transfer 
entropy computation. The estimators accept this information in the form of a `TEVars` 
instance, which can be constructed like so:

```julia
vars = TEVars(Tf = [1], Tpp = [2, 3], Spp = [4])
```

#### 4. Rectangular grid specification

Entropy is essentially a property of a collection of states. To meaningfully talk about 
states for our generalised state space reconstruction, we will divide the coordinate 
axes of the reconstruction into a rectangular grid. Each box in the grid will 
be considered a state, and the probability of visitation is equally distributed 
within the box. 

In this example, we'll use a rectangular partition where the box sizes are 
determined by splitting each coordinate axis into 6 equally spaced 
intervals, spanning the range of the data.

```julia 
binning = RectangularBinning(6)
```

#### 5. Compute transfer entropy

Now we're ready to compute transfer entropy. First, let's use the 
[`VisitationFrequency`](@ref) estimator with logarithm to the base 2. This 
gives the transfer entropy in units of bits.

```julia
estimator = VisitationFrequency(b = 2)
te_vf = transferentropy(embedding_pts, vars, binning, estimator) #, or
```

Okay, but what if we want to use another estimator and want the transfer 
entropy in units of nats? Easy. 

```julia
estimator = TransferOperatorGrid(b = Base.MathConstants.e)
transferentropy(embedding_pts, vars, binning, estimator)
```

Above, we computed transfer entropy for one particular choice of partition. 
Transfer entropy is a function of the  partition, and care must be taken with 
the choice of partition. Below is an example where we compute transfer entropy 
over 15 different cubic grids spanning the range of the data, with differing box sizes 
all having fixed edge lengths  (logarithmically spaced from 0.001 to 0.3).

```julia
# Define estimator
est = VisitationFrequency(b = 2)

# Define binning schemes based on different box sizes
edgelengths = 10 .^ range(log(10, 0.001), log10(0.3), length = 15)
bs = [RectangularBinning(ϵ) for ϵ in edgelengts]

tes = map(b -> transferentropy(embedding_pts, vars, b, est), bs)
```

`tes` now contains 15 different values of the transfer entropy, one for each of 
the discretization schemes. For the smallest bin sizes, the transfer entropy 
is close to or equal to zero, because there are not enough points distributed 
among the bins (of which there are many when the box edge length is small).
"""
function transferentropy(pts, vars::TEVars, ϵ::RectangularBinning, 
    estimator::BinningTransferEntropyEstimator) end


function transferentropy(pts, vars::TEVars, ϵ::RectangularBinning, 
        estimator::VisitationFrequency)

    # Base of the logarithm
    b = estimator.b

    # Collect variables for the marginals 
    C = vars.conditioned_presentpast
    XY = [vars.target_future;      vars.target_presentpast; C]
    YZ = [vars.target_presentpast; vars.source_presentpast; C]
    Y =  [vars.target_presentpast;                          C]
    
    # Find the bins visited by the joint system (and then get 
    # the marginal visits from that, so we don't have to encode 
    # bins multiple times). 
    joint_bin_visits = joint_visits(pts, ϵ)

    # Compute visitation frequencies for nonempty bi
    p_Y = non0hist(marginal_visits(joint_bin_visits, Y))
    p_XY = non0hist(marginal_visits(joint_bin_visits, XY))
    p_YZ = non0hist(marginal_visits(joint_bin_visits, YZ))
    p_joint = non0hist(joint_bin_visits)
    
    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(p_joint, b)
end

function transferentropy(pts, vars::TEVars, ϵ::RectangularBinning, 
        estimator::TransferOperatorGrid)
    
    # Base of the logarithm
    b = estimator.b

    # Collect variables for the marginals 
    C = vars.conditioned_presentpast
    XY = [vars.target_future;      vars.target_presentpast; C]
    YZ = [vars.target_presentpast; vars.source_presentpast; C]
    Y =  [vars.target_presentpast;                          C]

    # Calculate the invariant distribution over the bins.
    μ = invariantmeasure(pts, ϵ)
    
    # Find the unique visited bins, then find the subset of those bins 
    # with nonzero measure.
    positive_measure_bins = unique(μ.encoded_points)[μ.measure.nonzero_inds]

    p_Y  = marginal(Y, Array(transpose(hcat(positive_measure_bins...,))), μ.measure)
    p_XY = marginal(XY, Array(transpose(hcat(positive_measure_bins...,))), μ.measure)
    p_YZ = marginal(YZ, Array(transpose(hcat(positive_measure_bins...,))), μ.measure)

    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(μ.measure.dist[μ.measure.nonzero_inds], b)
end


# Allow CustomReconstructions to be used. 
transferentropy(pts::CustomReconstruction, vars::TEVars, ϵ::RectangularBinning, 
    estimator::VisitationFrequency) = 
    transferentropy(pts.reconstructed_pts, vars, ϵ, estimator)

    
transferentropy(pts::CustomReconstruction, vars::TEVars, ϵ::RectangularBinning, 
    estimator::TransferOperatorGrid) = 
    transferentropy(pts.reconstructed_pts, vars, ϵ, estimator)
