import CausalityToolsBase: RectangularBinning
import Distances: Metric, Chebyshev


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
- Transfer operator grid estimator. The symbols `:transferoperator_grid`, 
    `:transferoperatorgrid`, `:to_grid`, `:to_grid` will work.
- k nearest neighbours estimator. The symbols `:nn`, `:NN`, 
    `:nearestneighbors`, `:nearestneighbours`, `:nearest_neighbors`, 
    `:nearest_neighbours` will work.

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
        vars::TEVars, estimator::Symbol = :visitfreq; 
        k1::Int = 2, k2::Int = 3, metric::Metric = Chebyshev(), kwargs...)
    if estimator ∈ [:visitfreq, :visit_freq, :vf, :visitation_frequency, :visitationfrequency]
        transferentropy(pts, binning_scheme, vars, VisitationFrequency(); kwargs...)
    elseif estimator ∈ [:transferoperator_grid, :transferoperatorgrid, :to_grid, :to_grid]
        transferentropy(pts, binning_scheme, vars, TransferOperatorGrid(); kwargs...)
    elseif estimator ∈ [:nn, :NN, :nearestneighbors, :nearestneighbours, 
        :nearest_neighbors, :nearest_neighbours]
        transferentropy(pts, binning_scheme, vars, NNEstimator(), 
            k1 = k2, k2 = k2, metric = metric, kwargs...)
    end
end





