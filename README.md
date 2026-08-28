# FluxKAN.jl

[![Build Status](https://github.com/cometscome/FluxKAN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/FluxKAN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://github.com/cometscome/FluxKAN.jl/actions/workflows/Documentation.yml/badge.svg?branch=main)](https://cometscome.github.io/FluxKAN.jl/)
[![codecov](https://codecov.io/gh/cometscome/FluxKAN.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cometscome/FluxKAN.jl)

FluxKAN.jl provides Flux and Lux implementations of B-spline, polynomial, and
Gaussian-RBF Kolmogorov-Arnold Network (KAN) layers. `KANLinear` follows the
vectorized EfficientKAN formulation, the polynomial layers follow TorchKAN,
and the Gaussian layers follow FastKAN. See
[the research comparison](docs/research-comparison.md) for precise scope and
remaining differences from pykan.

## Layers

| Basis | Flux | Lux | Notes |
| --- | --- | --- | --- |
| B-spline | `KANLinear` | `LuxKANLinear` | EfficientKAN-style edge splines, residual branch, adaptive grids, and regularization |
| Legendre polynomial | `KALnet` | `LuxKALnet` | TorchKAN-style base branch, LayerNorm, and SiLU |
| Chebyshev polynomial | `KACnet` | `LuxKACnet` | TorchKAN-style KAC variant; not the base-free ChebyKAN layer |
| Gaussian RBF | `KAGnet` | `LuxKAGnet` | Fixed grid; FastKAN-style input LayerNorm and base branch |
| Gaussian RBF | `KAGLnet` | `LuxKAGLnet` | Experimental trainable grid centers |

All layers use feature-first arrays: the first dimension must equal `in_dim`,
and any remaining dimensions are treated as sample dimensions.

## Installation

```julia
using Pkg
Pkg.add("FluxKAN")
```

Lux users also need Lux itself:

```julia
Pkg.add("Lux")
```

## Flux usage

The B-spline layer is the closest implementation to the original KAN
architecture:

```julia
using Flux
using FluxKAN

model = Flux.Chain(
    KANLinear(2, 10; grid_size=5, spline_order=3),
    KANLinear(10, 1; grid_size=5, spline_order=3),
)

x = rand(Float32, 2, 32)
y = model(x)
```

Each edge learns `grid_size + spline_order` B-spline coefficients. The default
grid covers `[-1, 1]` and is extended by `spline_order` knots on each side.
Between optimizer steps, a layer grid can be adapted to representative CPU
samples while approximately preserving its spline functions:

```julia
update_grid!(model[1], x; margin=0.01)
penalty = kan_regularization(model[1])
```

Polynomial layers remain available:

```julia
using Flux
using FluxKAN

model = Flux.Chain(
    KALnet(2, 10; polynomial_order=3),
    KALnet(10, 1; polynomial_order=3),
)

x = rand(Float32, 2, 32)
```

Replace `KALnet` with `KACnet`, or use a Gaussian layer:

```julia
model = Flux.Chain(
    KAGnet(2, 10; num_grids=8),
    KAGnet(10, 1; num_grids=8),
)
```

`KAGnet` uses fixed grid centers. `KAGLnet` makes the centers trainable and is
experimental. In both layers the default grid is `-2:2`, matching FastKAN's
default range.

## Lux usage

The Lux variants are explicit-parameter `LuxCore.AbstractLuxLayer`s and can be
composed with normal Lux layers.

```julia
using FluxKAN
using Lux
using Random

rng = Random.Xoshiro(123)
model = Lux.Chain(
    LuxKANLinear(2, 10; grid_size=5, spline_order=3),
    LuxKANLinear(10, 1; grid_size=5, spline_order=3),
)

parameters, state = Lux.setup(rng, model)
x = rand(rng, Float32, 2, 32)
y, state = Lux.apply(model, x, parameters, state)
```

All Lux variants have the same architectural keywords as their Flux
counterparts. A B-spline grid is Lux state and can be adapted together with its
coefficients using `parameters, state = update_grid(layer, x, parameters,
state)`. Fixed RBF grids are state; learnable KAGL grids are parameters.

## MNIST example

```julia
using FluxKAN
using MLDatasets

result = FluxKAN.mnist_kan(
    batch_size=256,
    epochs=20,
    nhidden=64,
    grid_size=5,
    spline_order=3,
    method="BSpline",
)
```

Supported methods are `"BSpline"`, `"Legendre"`, `"Chebyshev"`, `"Gaussian"`,
and `"GaussianLearnable"`. The function returns the trained model and final
train and test metrics as a named tuple. `MLDatasets` is an optional dependency,
so install it separately before using this example.

## Correctness and compatibility

Version 0.9 targets Julia 1.10 or newer, Flux 0.16.11 or newer within the 0.16
series, LuxCore 1.3 or newer, and Lux 1 for the
test suite. Compared with version 0.1:

- polynomial ChainRules derivatives are correct for every tested order;
- batch-wide min-max normalization was removed, so constant inputs remain
  finite and one sample is unaffected by unrelated batch members;
- polynomial inputs use pointwise `tanh`, keeping Chebyshev and Legendre
  recurrence inputs bounded without batch-dependent inference;
- Gaussian layers now use the FastKAN computation order: input LayerNorm for
  the RBF branch, SiLU for the base branch, then addition without a post-sum
  activation;
- layer fields are concretely typed and Flux 0.16's explicit optimizer API is
  used by the example;
- constructor arguments and input dimensions are validated;
- a Cox-de Boor B-spline layer, adaptive per-feature grids, coefficient
  refitting, and EfficientKAN-style regularization are available for Flux and
  Lux.

The tests cover an independent EfficientKAN fixture, basis values,
finite-difference gradients, Flux/Lux numerical agreement, parameter gradients,
constant inputs, batch independence, type inference, grid-function
preservation, optimizer steps across a grid update, same-version serialization,
and end-to-end nonlinear B-spline regression convergence. Aqua and targeted JET
checks run separately in CI.

The feature-first array contract, B-spline boundary convention, stable API
candidate, and experimental API are documented in
[the API contract](docs/api.md). Changes from the registered 0.1 release are
listed in [the migration guide](docs/migration.md) and [changelog](CHANGELOG.md).

## Scope relative to the original KAN

`KANLinear` implements the central numerical mechanism of the original KAN:
learned B-spline edge functions, a residual base function, summation at nodes,
and adaptive per-feature knot grids with coefficient refitting. Its vectorized
parameterization and regularizer follow EfficientKAN. It does not yet include
pykan's masks, sparsification workflow, pruning, attribution, plotting, or
symbolic extraction, so it is not a full replacement for pykan's
interpretability toolchain.

## References

- Z. Liu et al., [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756), 2024.
- [pykan, the reference implementation](https://github.com/KindXiaoming/pykan).
- [EfficientKAN](https://github.com/Blealtan/efficient-kan).
- Z. Li, [Kolmogorov-Arnold Networks are Radial Basis Function Networks](https://arxiv.org/abs/2405.06721), 2024, and [FastKAN](https://github.com/ZiyaoLi/fast-kan).
- S. S. Bhattacharjee, [TorchKAN](https://github.com/1ssb/torchkan).
- S. S. SS et al., [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07200), 2024, and [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN).

## Citation

```bibtex
@misc{fluxkan,
  author = {Yuki Nagai},
  title = {FluxKAN.jl: Flux and Lux implementations of KAN variants},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/cometscome/FluxKAN.jl}}
}
```

## Author and support

Yuki Nagai, Ph.D., Associate Professor, Information Technology Center,
The University of Tokyo.

For support, open a GitHub issue or contact nagai.yuki@mail.u-tokyo.ac.jp.
