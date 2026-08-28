# Research and implementation comparison

This document records which published KAN variant each FluxKAN layer implements
and where the behavior intentionally differs. It is meant to prevent the broad
term "KAN" from hiding materially different architectures.

## Architectural comparison

| Work | Edge/basis model | Base branch | Normalization and output | Adaptive grid | Interpretability tools |
| --- | --- | --- | --- | --- | --- |
| Original KAN / pykan | Learnable univariate B-spline functions on edges; nodes sum incoming edge values | Residual base function plus spline function | No generic batch-dependent input min-max step | Yes | Sparsification, pruning, attribution, and symbolic extraction |
| EfficientKAN | Vectorized B-spline basis followed by spline weights | SiLU followed by a linear base weight | No post-sum activation in `KANLinear` | Yes, through `update_grid` | Regularization helper; fewer symbolic tools than pykan |
| FastKAN | Gaussian RBFs used to approximate the cubic B-spline basis | SiLU followed by a learned linear map | Optional input LayerNorm; base and RBF branches are summed directly | Fixed evenly spaced centers in the reference implementation | No pykan-style symbolic toolchain |
| ChebyKAN | Chebyshev polynomial basis after pointwise `tanh` | None in the reference layer | No internal output activation; the example network applies LayerNorm between layers | Not applicable | No pykan-style symbolic toolchain |
| TorchKAN KAL/KAC | Legendre or Chebyshev polynomial expansion followed by learned weights | SiLU followed by a learned linear map | TorchKAN code uses batch-wide min-max input scaling and output LayerNorm + SiLU | Not applicable | No pykan-style symbolic toolchain |
| FluxKAN 0.9 `KANLinear` | Cox-de Boor B-spline basis with learned coefficients on every input-output edge | SiLU followed by a bias-free learned linear map | Direct sum, with no post-sum activation | Yes; quantile/uniform interpolation followed by least-squares coefficient refitting | Coefficient regularization; no pruning or symbolic tools yet |
| FluxKAN 0.9 KAL/KAC | Legendre or Chebyshev polynomial expansion followed by learned weights | SiLU followed by a bias-free learned linear map | Pointwise `tanh` input bounding; output LayerNorm + SiLU | Not applicable | Not yet |
| FluxKAN 0.9 KAG | Gaussian RBF expansion followed by bias-free learned weights | SiLU followed by a learned affine map | FastKAN-style optional input LayerNorm; direct branch sum | Fixed evenly spaced centers | Not yet |
| FluxKAN 0.9 KAGL | Same as KAG | Optional | Same as KAG | Centers are trainable; width stays at its initial spacing | Not yet |

## Findings from the code comparison

### KANLinear

`KANLinear` implements the vectorized EfficientKAN data flow and the central
B-spline mechanism of the original KAN:

1. extend an evenly spaced knot grid by the spline order at both boundaries;
2. evaluate `grid_size + spline_order` Cox-de Boor basis functions per input;
3. learn separate spline coefficients for every input-output edge;
4. add the spline contribution to a bias-free SiLU residual branch;
5. optionally multiply every edge's spline coefficients by a learned scale;
6. adapt each feature's grid from sample quantiles and a uniform grid, then
   refit coefficients by least squares to approximately preserve edge curves.

Initialization also follows EfficientKAN: a small random curve is sampled at
the core grid points and converted to spline coefficients by least squares;
the residual weights and optional standalone spline scales use its Kaiming
uniform convention.

Both Flux and Lux implementations share the same basis evaluator. The Flux
grid is non-trainable layer data; the Lux grid is state. Adaptive grid updates
are explicit CPU operations between optimizer steps rather than differentiated
parts of the forward pass.

### KALnet and KACnet

These are TorchKAN-family layers, not direct implementations of the original
B-spline KAN. Their two-branch design and output LayerNorm + SiLU follow
TorchKAN's KAL/KAC structure. FluxKAN intentionally replaces TorchKAN's global
batch min-max scaling with pointwise `tanh`:

- a constant tensor cannot produce division-by-zero NaNs;
- evaluation of one sample does not change when other samples are added to its
  batch;
- polynomial recurrence inputs stay bounded.

The Chebyshev basis itself now matches ChebyKAN's pointwise `tanh` domain
mapping, but the surrounding KAC architecture still has a base branch and a
post-sum normalization/activation that the reference ChebyKAN layer does not.

### KAGnet

KAGnet is aligned with FastKAN's layer-level data flow:

1. optionally apply LayerNorm to the input of the RBF branch;
2. evaluate Gaussian RBF features on an evenly spaced `[-2, 2]` grid;
3. apply a bias-free linear map to those features;
4. apply SiLU and an affine map to the unnormalized input for the base branch;
5. sum the two branches without an additional activation.

This differs from FluxKAN 0.1, which normalized with extrema from the current
batch, used different branch ordering, and applied an output activation.

### KAGLnet

KAGLnet is a FluxKAN experiment, not a layer taken directly from FastKAN. Its
RBF centers are optimized as model parameters. The RBF denominator remains the
initial grid spacing, so center ordering and spacing are not constrained during
training. A robust future version should consider ordered parameterizations,
width learning, or an explicit grid-update procedure.

## Remaining pykan features

The numerical B-spline layer is now implemented. Matching the complete pykan
workflow still requires:

1. activation masks and a sparsification training workflow;
2. pruning and attribution APIs;
3. plotting individual learned edge functions;
4. symbolic fitting and extraction;
5. direct cross-language fixtures against pykan/EfficientKAN weights and grids;
6. GPU validation of the basis evaluator and a GPU-compatible grid-update path.

These are interpretability and tooling gaps, not missing B-spline forward or
gradient functionality.

## Primary sources

- [KAN paper](https://arxiv.org/abs/2404.19756) and [pykan](https://github.com/KindXiaoming/pykan)
- [EfficientKAN implementation](https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py)
- [FastKAN paper](https://arxiv.org/abs/2405.06721) and [implementation](https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py)
- [ChebyKAN paper](https://arxiv.org/abs/2405.07200) and [implementation](https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py)
- [TorchKAN KAL implementation](https://github.com/1ssb/torchkan/blob/main/KALnet.py) and [KAC implementation](https://github.com/1ssb/torchkan/blob/main/KACnet.py)
