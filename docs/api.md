# API contract

## Stable candidates for 1.0

The following names form the v0.9 public API candidate. Concrete fields of
layer structs are implementation details; construct and call layers through
their documented APIs.

```@docs
KANLinear
KALnet
KACnet
KAGnet
LuxKANLinear
LuxKALnet
LuxKACnet
LuxKAGnet
update_grid!
update_grid
kan_regularization
compute_legendre_polynomials
compute_chebyshev_polynomials
gaussian_rbf_basis
bspline_basis
mnist_kan
MNIST_KAN
```

## Array layout

All FluxKAN layers use feature-first arrays. `size(x, 1)` must equal `in_dim`;
all remaining dimensions are sample dimensions and are retained in the output.
A vector therefore denotes one sample and a matrix uses `(features, batch)`.

## B-spline boundaries

`bspline_basis` follows EfficientKAN's Cox-de Boor convention. Degree-zero
knot intervals are left-closed and right-open: `[kᵢ, kᵢ₊₁)`. Consequently,
all spline basis values are zero outside the complete knot support, including
at its final knot. `KANLinear` still has its residual/base branch outside the
spline support.

The default logical grid range is `[-1, 1]`; the stored knot vector is extended
by `spline_order` uniform knots on both sides. Values at the logical endpoint
`1` therefore remain inside the stored support for positive spline orders.

## Adaptive grids

Flux uses the mutating `update_grid!`; Lux uses the functional `update_grid`
and returns new parameters and state. Updates require CPU input with at least
`grid_size + spline_order` samples. Run them between optimizer steps. Existing
optimizer moments are retained, so applications making large or frequent grid
changes may choose to rebuild optimizer state afterward.

## Experimental API

```@docs
KAGLnet
LuxKAGLnet
```

The learnable-center Gaussian layers above are exported for experimentation but
are excluded from the planned 1.0 stability guarantee. Their center ordering
and width parameterization may change.

`MNIST_KAN` remains as a deprecated compatibility alias for `mnist_kan` and is
not part of the planned 1.0 API.
