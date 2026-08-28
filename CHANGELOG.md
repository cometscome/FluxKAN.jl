# Changelog

All notable changes to FluxKAN.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-08-28

### Added

- Flux and LuxCore `KANLinear` implementations with Cox-de Boor B-spline bases.
- EfficientKAN-compatible curve-to-coefficient and Kaiming initialization.
- Adaptive per-feature knot grids with least-squares coefficient refitting.
- EfficientKAN-style coefficient magnitude and entropy regularization.
- LuxCore versions of all existing layer families.
- Reproducible `rng` keywords for Flux layer constructors.
- Optional MLDatasets package extension and `mnist_kan` entry point.
- Independent EfficientKAN reference fixture, convergence, grid-update,
  serialization, Aqua, JET, multi-platform CI, and lower-bound checks.
- Versioned Documenter configuration and a reproducible benchmark suite.

### Changed

- Julia compatibility now starts at 1.10; Flux compatibility now starts at
  0.16.11 and LuxCore compatibility at 1.3.
- Polynomial recurrence inputs use pointwise `tanh`, eliminating batch-dependent
  normalization and constant-input NaNs.
- Gaussian layers follow FastKAN's normalization and branch ordering.
- Constructors and input dimensions are validated.
- `MNIST_KAN` is deprecated in favor of `mnist_kan`.

### Fixed

- Legendre and Chebyshev custom derivatives for every supported polynomial
  order.
- Layer behavior for scalar outputs, vectors, constant batches, and explicit
  Flux 0.16 optimizer updates.

### Removed from the public API

- Internal `Radial_distribution_function` and
  `Radial_distribution_function_L` types are no longer exported.

## [0.1.0] - 2025-05-02

- Initial registered release.
