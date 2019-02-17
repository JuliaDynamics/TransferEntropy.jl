import PerronFrobenius: AbstractTriangulationInvariantMeasure
import CausalityToolsBase: RectangularBinning
import StateSpaceReconstruction: Simplex, generate_interior_points
import StaticArrays: SVector


abstract type TransferEntropyEstimator end 

struct TransferOperatorGrid <: TransferEntropyEstimator end
struct VisitationFrequency <: TransferEntropyEstimator end


export transferentropy 

"""
    transferentropy(pts, binning_scheme::RectangularBinning, 
        vars::TEVars, estimator::Symbol = :visitfreq)

#### Transfer entropy using a rectangular partition

Estimate transfer entropy on a rectangular partition over 
a set of points `pts`, which represent a generalised embedding of 
data series ``x``, ``y`` and (potentially) ``z`` of the form 
outlined below. 

Note: `pts` must be a vector of states, not a vector of 
variables/(time series). Wrap your time series in a `Dataset`
first if the latter is the case.

#### Relationship between `pts` and `vars`

`pts` should be an embedding of the form 
``(y(t + \\eta)^{k}, y(t)^{l}, x(t)^{m}, z(t)^{n}``. 
Here, ``y(t + \\eta)^{k})`` indicates that ``k`` future states of `y` 
should be included, ``y(t)^{l}`` indicates that ``l`` present/past 
states of ``y`` should be included, ``x(t)^{m}`` indicates that ``m`` 
present/past states of ``x`` should be included, and ``z(t)^{n}`` indicates 
that ``n`` present/past states of the variable we're conditioning on should 
be included in the embedding vectors. Thus, the total dimension 
of the embedding space will be ``k + l + m + n``. 

`vars` is a `TEVars` instance contain the instruction on which 
variables of the embedding will be treated as part of which marginals
during transfer entropy computation. Check the documentation of 
`TEVars` for more details.

#### Estimators (and their acronyms)

- Visitation frequency estimator. The symbols `:visitfreq`, `:vf` 
    and `:visitation_frequency` will work.


### Example 

#### 1. Time series

We'll generate two 80-point long realizations, ``x`` and ``y``, of two 1D 
logistic maps starting at different initial conditions.

```julia
sys1 = DynamicalSystems.Systems.logistic()
sys2 = DynamicalSystems.Systems.logistic()
x = trajectory(sys1, 80, Ttr = 1000);
y = trajectory(sys1, 80, Ttr = 1000);

# Wrap the time series in a dataset containing the states of the 
# composite system.
pts = Dataset(x, y)
```

#### 2. Generalised embedding
Say we want to compute transfer entropy from ``x`` to ``y``, and that we 
require a 5-dimensional embedding. We'll choose a generalised embedding 
of the form ``((y(t + \\eta), y(t), y(t - \\tau_1), y(t - \\tau_2), x(t))`` 
with ``\\tau_1 = 1`` and ``\\tau_2 = 4``. These lags may be chosen, for example, 
as the first two minima of the autocorrelation or lagged mutual information 
function between the time series. 

To create the embedding, we'll use the `customembed` function (check its 
documentation for a detailed explanation on how it works). 

```julia
# Wrap the time series in a dataset, so we can easily index the state vectors.
data = Dataset(x, y);

# Embed the data, putting time series in the 2nd column (y) of `data` in the 
# first four embedding columns, lagging them with lags (1, 0, -1, -4), and 
# putting the 1st column of `data` (x) in the last position of the embedding,
# not lagging it.
embedding = customembed(data, Positions(2, 2, 2, 2, 1), Lags(1, 0, -1, -4, 0));
```

#### 3. Instructions to the estimator

Now, tell the estimator how to relative the dynamical variables of the 
generalised embedding to the marginals in the transfer entropy computation.

```julia
vars = TEVars(target_future = [1], 
    target_presentpast = [2, 3, 4], 
    source_presentpast = [5])
```

#### 4. Rectangular grid specification

We'll compute transfer entropy using the visitation frequency estimator over 
a rectangular partition where the box sizes are determined by 
splitting each coordinate axis into ``20`` equally spaced intervals each.

```julia 
binning = RectangularBinning(20)
```

#### 5. Compute transfer entropy

```julia
# Over a single rectangular grid
transferentropy(pts, binning, vars, :visitfreq)

# Over multiple cubic grids with differing box sizes
# logarithmically spaced from edge length 0.001 to 0.3
ϵs = 10 .^ range(log(10, 0.001), log10(0.3), length = 15)
map(ϵ -> transferentropy(pts, RectangularBinning(ϵ), vars, :visitfreq))
```
"""
function transferentropy(pts, binning_scheme::RectangularBinning, 
        vars::TEVars, estimator::Symbol = :visitfreq)
    if estimator ∈ [:visitfreq, :vf, :visitation_frequency]
        transferentropy_visitfreq(pts, binning_scheme, vars)
    end
end

"""
    transferentropy(μ::AbstractTriangulationInvariantMeasure, 
        binning_scheme::RectangularBinning, vars::TEVars; n::Int = 10000)

#### Transfer entropy using a precomputed invariant measure over a triangulated partition

Estimate transfer entropy from an invariant measure over a triangulation
that has been precomputed either as 

1. `μ = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())`, or
2. `μ = invariantmeasure(pts, TriangulationBinning(), ExactIntersection())` 

where the first method uses approximate simplex intersections (faster) and 
the second method uses exact simplex intersections (slow). `μ` contains 
all the information needed to compute transfer entropy. 

Note: `pts` must be a vector of states, not a vector of 
variables/(time series). Wrap your time series in a `Dataset`
first if the latter is the case.

#### Computing transfer entropy (triangulation -> rectangular partition)

Because we need to compute marginals, we need a rectangular grid. To do so,
transfer entropy is computed by sampling the simplices of the 
triangulation according to their measure with a total of approximately 
`n` points. Introducing multiple points as representatives for the partition
elements does not introduce any bias, because we in computing the 
invariant measure, we use no more information than what is encoded in the 
dynamics of the original data points. However, from the invariant measure,
we can get a practically infinite amount of points to estimate transfer 
entropy from.

Then, transfer entropy is estimated using the visitation 
frequency estimator on those points (see docs for `transferentropy_visitfreq` 
for more information), on a rectangular grid specified by `binning_scheme`.

#### Common use case

This method is good to use if you want to explore the sensitivity 
of transfer entropy to the bin size in the final rectangular grid, 
when you have few observations in the time series. The invariant 
measure, which encodes the dynamical information, is slow to compute over 
the triangulation, but only needs to be computed once.
After that, transfer entropy may be estimated at multiple scales very quickly.

### Example 

```julia
# Compute invariant measure over a triangulation using approximate 
# simplex intersections. This is relatively slow.
μ = invariantmeasure(pts, TriangulationBinning(), ApproximateIntersection())

# Compute transfer entropy from the invariant measure over multiple 
# bin sizes. This is fast, because the measure has been precomputed.
tes = map(ϵ -> transferentropy(μ, RectangularBinning(ϵ), TEVars([1], [2], [3])), 2:50)
```
"""
function transferentropy(μ::AbstractTriangulationInvariantMeasure, 
        binning_scheme::RectangularBinning, vars::TEVars; n::Int = 10000)
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
    fillpts = Vector{SVector{dim, Float64}}(undef, length(n_fillpts_persimplex))

    for i = 1:n_simplices
        sᵢ = simplices[i]
        if n_fillpts_persimplex[i] > 0
            pts = generate_interior_points(sᵢ, n_fillpts_persimplex[i])
            append!(fillpts, [SVector{dim, Float64}(pt) for pt in pts])
        end
    end
    
    transferentropy_visitfreq(fillpts, binning_scheme, vars)
end



"""
    transferentropy(pts, ϵ, vars::TEVars, estimator::VisitationFrequency; b = 2)

Compute transfer entropy for a set of ordered points representing
an appropriate embedding of some time series. See documentation for 
`TEVars` for info on how to specify the marginals (i.e. which variables 
of the embedding are treated as what). 

`b` sets the base of the logarithm (e.g `b = 2` gives the transfer 
entropy in bits). 
"""
function transferentropy(pts, ϵ, vars::TEVars, estimator::VisitationFrequency; b = 2)
    
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


"""
    transferentropy(pts, ϵ, vars::TEVars, estimator::TransferOperatorGrid; b = 2)

Compute transfer entropy for a set of ordered points representing
an appropriate embedding of some time series. See documentation for 
`TEVars` for info on how to specify the marginals (i.e. which variables 
of the embedding are treated as what). 

`b` sets the base of the logarithm (e.g `b = 2` gives the transfer 
entropy in bits). 
"""
function transferentropy(pts::Vector{T}, ϵ, vars::TEVars, estimator::TransferOperatorGrid; 
        b = 2) where {T <: Vector, SVector, MVector}
    

    # Calculate the invariant distribution over the bins.
    μ = invariantmeasure(pts, ϵ)
    
    # Find the unique visited bins, then find the subset of those bins 
    # with nonzero measure.
    # Make a version of marginal that takes columns 
    positive_measure_bins = transpose(unique(μ.visited_bins_inds, dims = 2))[μ.measure.nonzero_inds, :]
    
        # Collect variables for the marginals 
    C = vars.conditioned_presentpast
    XY = [vars.target_future;      vars.target_presentpast; C]
    YZ = [vars.target_presentpast; vars.source_presentpast; C]
    Y =  [vars.target_presentpast;                          C]

    p_Y  = marginal(Y, positive_measure_bins, μ.measure)
    p_XY = marginal(XY, positive_measure_bins, μ.measure)
    p_YZ = marginal(YZ, positive_measure_bins, μ.measure)
    
    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(μ.measure.dist[μ.measure.nonzero_inds], b)
end