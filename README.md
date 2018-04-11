# TransferEntropy.jl

Julia package for computing transfer entropy (TE). 

## Estimators
Only one estimator is currently implemented.

### Transfer Operator Transfer Entropy (TOTE)
Transfer entropy estimators traditionally derive state probabilities from counting. TOTE, in contrast, is based on a numerical approximation of the transfer operator of the underlying dynamics. 

In short, the underlying dynamics is replaced by a stationary Markov process. From this Markov process, an invariant probaiblity distribution on the state space can be derived. Transfer entropy is then estimated from this invariant probability distribution. 

## Installation
TransferEntropy.jl relies on several sobroutines implemented in other packages. Before these become registered Julia packages, you will have to install dependencies manually.  Running the following in the Julia console should get you up and running. 

```
# Dependencies needed by the subroutines
Pkg.add("PyCall")
Pkg.add("Conda")
Pkg.add("Distributions")
Pkg.add("Parameters")
Pkg.add("ProgressMeter")
Pkg.add("ProgressMeter")
Pkg.add("PmapProgressMeter")

# Subroutines 
Pkg.clone("https://github.com/kahaaga/Simplices.jl")
Pkg.clone("https://github.com/kahaaga/SimplexSplitting.jl")
Pkg.clone("https://github.com/kahaaga/InvariantDistribution.jl")

# Finally, install the TransferEntropy.jl package.
Pkg.clone("https://github.com/kahaaga/TransferEntropy.jl")
```

## Usage 
Using the estimator is easy. Imagine you have two time series `x` and `y`, just run the following

```
# Load the TransferEntropy package
using TransferEntropy 

# Create some random time series to test on
n_pts = 50
x = rand(n_pts)
y = rand(n_pts)

#= 
Calculate transfer entropy with the default `te_lag = 1`. 

The first argument is the assumed source (driver) and the second argument is the assumed response. `te_from_timeseries` returns a tuple of information produced during the run. This includes a state space reconstruction of the observations, a (possibly refined) triangulation of the state space, the estimated transfer operator (Markov matrix), the invariant distribution on the simplices of the triangulation and, finally, the transfer entropy estimate.
=#
te_result = te_from_timeseries(x, y)
```
