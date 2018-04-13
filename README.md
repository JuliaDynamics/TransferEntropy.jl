# TransferEntropy.jl

Julia package for computing transfer entropy (TE). 

## Estimators

### 1. Transfer Operator Transfer Entropy (TOTE)
Transfer entropy estimators traditionally derive state probabilities from counting. TOTE, in contrast, is based on a numerical approximation of the transfer operator of the underlying dynamics. 

The estimation of the transfer operator is based on the classic works by Stanislaw Ulam, and more recently by Gary Froyland (<http://web.maths.unsw.edu.au/~froyland/>). A short background article describing the concept of the transfer operator from time series can be found at <https://www.earthsystemevolution.com/articles/20180324_estimating_ergodic_probability_distributions/>. 

In short, the underlying dynamics is replaced by a stationary Markov process. From this Markov process, an invariant probability distribution on the state space is computed. Transfer entropy is then estimated from this invariant probability distribution. 

## Installation
TransferEntropy.jl relies on several subroutines implemented in other packages. Until these become registered Julia packages, you will have to install the dependencies manually.  Entering the following commands in the Julia console should get you up and running. 

```
# Dependencies needed by the subroutines
Pkg.add("PyCall")
Pkg.add("Conda")
Pkg.add("Distributions")
Pkg.add("Parameters")
Pkg.add("ProgressMeter")
Pkg.add("ProgressMeter")
Pkg.add("PmapProgressMeter")
Pkg.add("Plots")
Pkg.add("PlotlyJS")

# Subroutines 
Pkg.clone("https://github.com/kahaaga/Simplices.jl")
Pkg.clone("https://github.com/kahaaga/SimplexSplitting.jl")
Pkg.clone("https://github.com/kahaaga/InvariantDistribution.jl")

# Finally, install the TransferEntropy.jl package.
Pkg.clone("https://github.com/kahaaga/TransferEntropy.jl")
```

## Usage 
Using the estimator is easy. The workhorse is the `te_from_ts` function. 

Imagine you have two time series `x` and `y`. To compute TE, run the following:

```
# Load the TransferEntropy package
using TransferEntropy 

# Create some random time series to test on
n_pts = 50
x = rand(n_pts)
y = rand(n_pts)

#= 
Calculate transfer entropy with the default `te_lag = 1`. 

The first argument is the assumed source (driver) and the second argument is the assumed response. `te_from_ts` returns a tuple of 
information produced during the run. This tuple contains the following:
[1] a state space reconstruction of the observations
[2] a (possibly refined) triangulation of the state space
[3] the estimated transfer operator (Markov matrix)
[4] the invariant distribution on the simplices of the triangulation
[5] a TEResult, containing the transfer entropy estimate.
=#

te_result = te_from_ts(x, y) 

```
### Plot the results 

```
TE = te_result[5]
using Plots; plotlyjs() 
plot(TE.binsizes, TE.TE) # All repetitions 
plot!(TE.binsizes, mean(TE.TE, 2), label = "Mean TE", lw = 3) # Plot the mean
```

### Specifying keyword arguments

```
##############################
# Specifying keyword arguments 
##############################

#= 
Changing the number of TE estimates (in how many ways do we substitute each simplex by a point representation?, 
higher numbers give more accurate estimates).
=#
te_result = te_from_ts(x, y, n_reps = 1) 

# Manually setting bin sizes
te_result = te_from_ts(x, y, binsizes = 10:10:100)
te_result = te_from_ts(x, y, binsizes = [2, 30, 31, 47])

# Adjusting the transfer entropy lag (method lag, not lag in underlying; default = 1)
te_result = te_from_ts(x, y, te_lag = 2)

# Refine triangulation
te_result = te_from_ts(x, y, refine = true)

# Discrete approximation (discrete is fast, but has errors of ~5% if ~100 pts is used)
te_result = te_from_ts(x, y, discrete = true, n_randpts::Int = 100, sample_uniformly = true)

# Exact approximation (much slower, but no bias introduced beyond what is present in data)
te_result = te_from_ts(x, y, discrete = false) 

# Exact approximation in parallel (start julia with `julia -p n_procs`, e.g. `julia -p 4`, or use `addprocs()`,
# then load the library like this: `@everywhere using TransferEntropy`)
te_result = te_from_ts(x, y, discrete = false, parallel = true) 
```

## Problems? Do you want to compute transfer entropy, but don't know how?  
I'll be happy to help with setting up analyses. Feel free to contact me at e-mail (`kahaaga@gmail.com` or `kristian.haaga@uib.no`). 

If you encounter problems, submit an issue or send an e-mail.
