# FluxKAN.jl

FluxKAN.jl provides Flux and LuxCore implementations of B-spline,
Legendre-polynomial, Chebyshev-polynomial, and Gaussian-RBF
Kolmogorov-Arnold Network layers.

## Installation

```julia
using Pkg
Pkg.add("FluxKAN")
```

The full `Lux` package is only needed when using Lux's setup, composition, and
training utilities. FluxKAN itself depends on the lighter `LuxCore` interface.

## Minimal Flux example

```julia
using Flux
using FluxKAN

model = Chain(KANLinear(2, 8), KANLinear(8, 1))
x = rand(Float32, 2, 32)
size(model(x)) == (1, 32)
```

See the [API contract](api.md) before relying on grid mutation or boundary
behavior. The [research comparison](research-comparison.md) distinguishes the implemented numerical layers
from pykan's broader interpretability workflow.
