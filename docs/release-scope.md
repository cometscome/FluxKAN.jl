# Version 1.0 scope

Version 1.0 will mean that FluxKAN's documented layer constructors, array
layout, forward semantics, B-spline boundary behavior, and grid-update APIs are
stable under semantic versioning.

The 1.0 scope is a reliable Flux/LuxCore numerical layer library. It includes:

- B-spline, Legendre, Chebyshev, and fixed Gaussian-RBF layers;
- differentiable forward passes and tested parameter gradients;
- adaptive B-spline grid refitting on CPU;
- coefficient regularization;
- deterministic initialization and same-version serialization tests;
- Julia 1.10/current-stable CI and package quality checks.

The following pykan workflow features are explicitly not 1.0 blockers:

- activation masks and sparsification schedules;
- pruning and attribution;
- learned-edge plotting;
- symbolic fitting and formula extraction;
- GPU adaptive-grid updates.

They may be added compatibly in 1.x. Learnable Gaussian centers (`KAGLnet` and
`LuxKAGLnet`) remain experimental until an ordered-center and width policy is
chosen.
