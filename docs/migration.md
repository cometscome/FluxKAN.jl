# Migration from 0.1

Version 0.9 is a stabilization release and intentionally changes behavior that
was incorrect or batch-dependent in 0.1.

## Required changes

- Flux 0.16.11 and Julia 1.10 or newer are required.
- Use Flux's explicit optimizer API: `state = Flux.setup(opt, model)` followed
  by `Flux.update!(state, model, gradients)`.
- `Radial_distribution_function` and `Radial_distribution_function_L` are
  internal implementation types and are no longer exported. Use `KAGnet`,
  `KAGLnet`, or `gaussian_rbf_basis`.
- Prefer `mnist_kan`; `MNIST_KAN` is retained as a deprecated alias during the
  0.9 transition.
- Inputs are explicitly feature-first. Matrices use `(features, batch)`.

## Numerical changes

- Legendre and Chebyshev ChainRules derivatives have been corrected.
- Polynomial inputs use pointwise `tanh` rather than batch-wide min/max
  normalization. Predictions no longer depend on unrelated batch members.
- Gaussian layers follow FastKAN's branch order and no longer apply an extra
  activation after summing branches.
- Constructor dimension and grid validation now fails early with descriptive
  errors.

## New APIs

- `KANLinear` and `LuxKANLinear` provide EfficientKAN-style B-spline layers.
- `update_grid!` and `update_grid` adapt knots and refit coefficients.
- `kan_regularization` provides coefficient-magnitude and entropy penalties.
- All Flux layer constructors accept an `rng` keyword for reproducible
  initialization.
