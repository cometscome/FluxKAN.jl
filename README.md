# FluxKAN.jl

[![Build Status](https://github.com/cometscome/FluxKAN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/FluxKAN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://github.com/cometscome/FluxKAN.jl/actions/workflows/Documentation.yml/badge.svg?branch=main)](https://cometscome.github.io/FluxKAN.jl/)
[![codecov](https://codecov.io/gh/cometscome/FluxKAN.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cometscome/FluxKAN.jl)

FluxKAN.jl provides Kolmogorov–Arnold Network (KAN) layers for both
[Flux.jl](https://fluxml.ai/Flux.jl/) and [Lux.jl](https://lux.csail.mit.edu/).
It includes a trainable B-spline layer based on EfficientKAN as well as
Legendre, Chebyshev, and Gaussian-RBF variants.

The current source tree targets **FluxKAN v0.9.0**. All layers use Julia's
feature-first convention: a matrix input has shape `(features, batch)`, and
`size(x, 1)` must equal the layer's input dimension.

## What's new in v0.9

Version 0.9 is a substantial update from the registered v0.1 release:

- **B-spline KAN:** `KANLinear` and `LuxKANLinear` implement Cox–de Boor
  B-spline edge functions, an EfficientKAN-style residual branch, curve-based
  initialization, optional per-edge spline scaling, and regularization.
- **Flux and Lux parity:** every layer family now has both a Flux layer and a
  LuxCore-compatible explicit-parameter layer.
- **Adaptive grids:** B-spline knots can be adapted per input feature and the
  spline coefficients are refitted by least squares to approximately preserve
  the learned functions.
- **Correctness fixes:** Legendre and Chebyshev derivatives were corrected;
  batch-dependent min–max normalization and constant-input NaNs were removed;
  Gaussian layers now follow FastKAN's branch ordering.
- **Reproducible and safer APIs:** Flux constructors accept `rng`, dimensions
  and keyword values are validated, and the MNIST helper is an optional
  MLDatasets extension.
- **Release engineering:** the test suite now covers independent EfficientKAN
  fixtures, finite-difference gradients, Flux/Lux agreement, grid updates,
  serialization, and nonlinear regression convergence. Aqua, JET,
  multi-platform CI, documentation, and benchmarks were also added.

See the [changelog](CHANGELOG.md) for the complete list and the
[migration guide](docs/migration.md) when upgrading from v0.1.

## Available layers

| Basis / architecture | Flux | Lux | Status |
| --- | --- | --- | --- |
| B-spline / EfficientKAN | `KANLinear` | `LuxKANLinear` | Recommended for a standard KAN |
| Legendre / TorchKAN KAL | `KALnet` | `LuxKALnet` | Supported |
| Chebyshev / TorchKAN KAC | `KACnet` | `LuxKACnet` | Supported |
| Gaussian RBF / FastKAN | `KAGnet` | `LuxKAGnet` | Supported; fixed centers |
| Gaussian RBF | `KAGLnet` | `LuxKAGLnet` | Experimental; trainable centers |

`KANLinear` is the closest layer in this package to the original B-spline KAN.
The polynomial and Gaussian layers are related KAN variants rather than
drop-in reproductions of pykan. The exact architectural correspondence is
documented in the [research comparison](docs/research-comparison.md).

## Installation

Install the registered package and Flux frontend with:

```julia
using Pkg
Pkg.add(["FluxKAN", "Flux"])
```

`Pkg.add` selects the latest registered release. If v0.9 has not yet reached
the General registry, use the development installation below to access the
features described in this README.

To try the development version directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/cometscome/FluxKAN.jl")
Pkg.add("Flux")
```

Flux support is available immediately. For Lux model composition and its
training API, install Lux and the training backend packages used in the
examples below:

```julia
using Pkg
Pkg.add(["Lux", "Optimisers", "Zygote"])
```

Compatibility for v0.9:

| Package | Supported versions |
| --- | --- |
| Julia | `1.10` or newer |
| Flux | `0.16.11` or newer within `0.16.x` |
| LuxCore | `1.3` or newer within `1.x` |
| Lux | `1.x` |

## Flux quick start

Build a B-spline KAN exactly like any other Flux model:

```julia
using Flux
using FluxKAN
using Random

rng = Random.Xoshiro(123)
model = Flux.Chain(
    KANLinear(2, 10; grid_size=5, spline_order=3, rng),
    KANLinear(10, 1; grid_size=5, spline_order=3, rng),
)

x = rand(rng, Float32, 2, 32)
y = model(x)                         # size (1, 32)
```

`grid_size` is the number of intervals in the logical grid and
`spline_order` is the polynomial degree. Each edge learns
`grid_size + spline_order` spline coefficients. The default logical range is
`[-1, 1]`; the stored knot vector is extended at both ends so splines remain
defined near the logical boundary.

### Training a B-spline layer with Flux

This small regression example is also exercised by the package test suite:

```julia
using Flux
using FluxKAN
using Random

x = reshape(collect(range(-0.95f0, 0.95f0; length=128)), 1, :)
target = sin.(3f0 * Float32(pi) .* x) .+ 0.25f0 .* x .^ 2

layer = KANLinear(
    1,
    1;
    grid_size=12,
    spline_order=3,
    standalone_spline_scale=false,
    rng=Random.Xoshiro(44),
)

loss(m) = Flux.mse(m(x), target)
opt_state = Flux.setup(Flux.Adam(0.03), layer)

for _ in 1:300
    grads = Flux.gradient(loss, layer)[1]
    Flux.update!(opt_state, layer, grads)
end

@show loss(layer)                    # tested to reach < 1e-3
```

The base and spline branches are trained together. To add the
EfficientKAN-style coefficient magnitude and entropy penalty, use for example:

```julia
loss(m) = Flux.mse(m(x), target) + 1f-4 * kan_regularization(m)
```

## Lux quick start

The Lux variants are native `LuxCore.AbstractLuxLayer`s. Their architecture is
immutable, parameters are explicit, and non-trainable data such as a B-spline
grid is stored in state. They compose with ordinary Lux layers:

```julia
using FluxKAN
using Lux
using Random

rng = Random.Xoshiro(123)
model = Lux.Chain(
    LuxKANLinear(2, 10; grid_size=5, spline_order=3),
    LuxKANLinear(10, 1; grid_size=5, spline_order=3),
)

ps, st = Lux.setup(rng, model)
x = rand(rng, Float32, 2, 32)
y, st = Lux.apply(model, x, ps, st)  # size (1, 32)
```

Every Flux layer has a corresponding Lux constructor with the same
architectural keywords:

```julia
LuxKALnet(2, 10; polynomial_order=3)
LuxKACnet(2, 10; polynomial_order=3)
LuxKAGnet(2, 10; num_grids=8)
LuxKAGLnet(2, 10; num_grids=8)       # experimental
```

For fixed-center `LuxKAGnet`, RBF centers live in state. For experimental
`LuxKAGLnet`, the centers are trainable parameters.

### Training a B-spline layer with Lux

Here is the corresponding regression using Lux's explicit training state:

```julia
using FluxKAN
using Lux
using Lux.Training
using Optimisers
using Random
using Zygote

x = reshape(collect(range(-0.95f0, 0.95f0; length=128)), 1, :)
target = sin.(3f0 * Float32(pi) .* x) .+ 0.25f0 .* x .^ 2

model = LuxKANLinear(
    1,
    1;
    grid_size=12,
    spline_order=3,
    standalone_spline_scale=false,
)
ps, st = Lux.setup(Random.Xoshiro(45), model)

function objective(model, ps, st, (x, target))
    prediction, st = Lux.apply(model, x, ps, st)
    loss = sum(abs2, prediction .- target) / length(target)
    return loss, st, (;)
end

function train(model, ps, st, data; steps=300)
    train_state = Training.TrainState(
        model, ps, st, Optimisers.Adam(0.03)
    )
    for _ in 1:steps
        _, loss, _, train_state = Training.single_train_step!(
            AutoZygote(), objective, data, train_state
        )
    end
    return train_state
end

train_state = train(model, ps, st, (x, target))
prediction, st = Lux.apply(
    model, x, train_state.parameters, train_state.states
)
@show sum(abs2, prediction .- target) / length(target)
```

For a standalone Lux B-spline layer, regularization is computed from explicit
parameters:

```julia
penalty = kan_regularization(model, train_state.parameters)
```

## Why LuxCore is a direct dependency

FluxKAN depends directly on the lightweight
[`LuxCore`](https://lux.csail.mit.edu/stable/manual/interface) package, not on
the full `Lux` package. This is intentional: the exported `Lux*` types subtype
`LuxCore.AbstractLuxLayer`, and their parameter/state interface must be defined
whenever FluxKAN is loaded. Lux itself is only needed when an application wants
containers such as `Lux.Chain`, device utilities, or Lux's training API, so it
remains an application/test dependency rather than a runtime dependency of
FluxKAN.

MLDatasets is different: it is a true weak dependency and only activates the
optional MNIST example extension when the user loads MLDatasets.

## Adaptive B-spline grids

The adaptive-grid operation combines sample quantiles with a uniform grid and
then refits the coefficients. It is an explicit operation intended to run
between optimizer steps on CPU, not a differentiable part of the forward pass.

Flux mutates the layer:

```julia
update_grid!(layer, x; margin=0.01)
```

Lux returns new parameters and state:

```julia
ps, st = update_grid(model, x, ps, st; margin=0.01)
```

For a multi-layer network, update layers in forward order and pass the current
activation samples to each layer. An update requires at least
`grid_size + spline_order` samples. Existing optimizer moments can be retained
for small occasional updates; rebuilding optimizer state may be preferable
after a large or frequent grid change.

## Polynomial and Gaussian variants

The other layer families use the same feature-first input layout:

```julia
using Flux
using FluxKAN

legendre_model = Flux.Chain(
    KALnet(2, 10; polynomial_order=3),
    KALnet(10, 1; polynomial_order=3),
)

chebyshev_model = Flux.Chain(
    KACnet(2, 10; polynomial_order=3),
    KACnet(10, 1; polynomial_order=3),
)

gaussian_model = Flux.Chain(
    KAGnet(2, 10; num_grids=8),
    KAGnet(10, 1; num_grids=8),
)
```

`KAGnet` uses fixed grid centers and is the recommended Gaussian variant.
`KAGLnet` makes the centers trainable and is currently experimental. The
default Gaussian grid covers `[-2, 2]`, matching FastKAN's default range.

## MNIST example

MLDatasets is optional, so install and load it explicitly:

```julia
using Pkg
Pkg.add("MLDatasets")

using FluxKAN
using MLDatasets

result = mnist_kan(
    batch_size=256,
    epochs=20,
    nhidden=64,
    grid_size=5,
    spline_order=3,
    method="BSpline",
)

@show result.train_accuracy result.test_accuracy
```

Supported methods are `"BSpline"`, `"Legendre"`, `"Chebyshev"`,
`"Gaussian"`, and `"GaussianLearnable"`. The return value contains the trained
Flux model and final train/test losses and accuracies. `MNIST_KAN` remains as a
deprecated compatibility alias for `mnist_kan`.

## Scope and current limitations

`KANLinear` implements the central numerical mechanism of the original KAN:
learned B-spline functions on edges, a residual base function, summation at
nodes, and adaptive per-feature grids with coefficient refitting. Its
vectorized parameterization and regularizer follow EfficientKAN.

FluxKAN v0.9 is a layer library, not yet a replacement for the complete pykan
interpretability workflow. It does not currently provide masks,
sparsification, pruning, attribution, learned-function plotting, or symbolic
extraction. GPU forward/backward support is not claimed for v0.9, and adaptive
grid updates are CPU-only. See the [release scope](docs/release-scope.md) and
[performance notes](docs/performance.md) for details.

## Correctness and testing

The v0.9 tests cover:

- B-spline degrees 0–4, partition of unity, boundaries, and finite-difference
  derivatives;
- an independent EfficientKAN basis and forward-pass fixture;
- Flux/Lux numerical parity and parameter gradients;
- constant inputs, batch independence, and input validation;
- function preservation and continued optimization across grid updates;
- end-to-end nonlinear B-spline regression convergence;
- reproducible initialization and same-version serialization.

Run the main test suite with:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Aqua and targeted JET checks run in a separate `qa/` environment. CI also
tests Julia 1.10, Julia 1.12, Julia nightly, Linux, macOS, Windows, and the
declared lower dependency bounds.

The public API contract and B-spline boundary convention are documented in
[the API reference](docs/api.md).

## References

- Z. Liu et al., [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756), 2024, and [pykan](https://github.com/KindXiaoming/pykan).
- [EfficientKAN](https://github.com/Blealtan/efficient-kan), the basis for `KANLinear` and `LuxKANLinear`.
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

For support, open a [GitHub issue](https://github.com/cometscome/FluxKAN.jl/issues)
or contact nagai.yuki@mail.u-tokyo.ac.jp.
