# TransferEntropy.jl

Julia package for computing transfer entropy (TE).

## Installation
Until the package is registered on METADATA, install it by running the following
line in the Julia console

```julia
Pkg.clone("https://github.com/kahaaga/TransferEntropy.jl")
```

TransferEntropy.jl relies on several subroutines implemented in other packages.
Until these become registered Julia packages, you will have to install the
dependencies manually.  Entering the following commands in the Julia console
should get you up and running.

```julia
using TransferEntropy
#Install dependencies that are not registered on METADATA yet
install_dependencies()
```
