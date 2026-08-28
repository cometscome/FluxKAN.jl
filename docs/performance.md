# Performance

B-spline layers trade compute and memory for learnable edge functions. For an
input with `I` features, `O` outputs, batch size `B`, grid size `G`, and spline
order `K`, the expanded basis contains approximately `I × (G + K) × B`
elements and the spline coefficients contain `O × I × (G + K)` elements.

The repository's `benchmark/` environment measures:

- `Dense`, `KANLinear`, and `KAGnet` Flux forward passes;
- Flux and Lux B-spline gradients;
- Lux forward passes;
- adaptive-grid coefficient refitting.

Run it with:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/runbenchmarks.jl benchmark-v0.9.json
```

Wall-clock regression thresholds are not enforced in CI because hosted runner
hardware varies. Keep the Julia version, BLAS implementation, CPU, input shape,
and thread count fixed when comparing saved results.

## v0.9 development baseline

The following minimum-of-five measurements use Julia 1.12.6 with one Julia
thread, OpenBLAS, an AMD EPYC 9554 CPU, and input shape `(32, 256)`. Outputs
have 64 features; `KANLinear` uses the default grid size 5/order 3 and `KAGnet`
uses 8 centers. These numbers document scale, not a portable performance
guarantee.

| Workload | Time | Allocated | Allocations |
| --- | ---: | ---: | ---: |
| Dense forward | 0.011 ms | 0.063 MiB | 3 |
| KAGnet forward | 0.570 ms | 1.03 MiB | 34 |
| KANLinear forward | 2.45 ms | 4.86 MiB | 114 |
| LuxKANLinear forward | 2.46 ms | 4.86 MiB | 114 |
| KANLinear gradient | 12.5 ms | 33.63 MiB | 4,812 |
| LuxKANLinear gradient | 9.02 ms | 33.62 MiB | 4,855 |
| `update_grid!` | 7.97 ms | 14.55 MiB | 1,716 |

The table makes the current optimization target explicit: B-spline basis and
gradient materialization dominate allocations. Future releases can use the
saved BenchmarkTools results to quantify improvements without changing API
semantics.

GPU forward and backward execution are not claimed as supported in v0.9.
Adaptive-grid updates are CPU-only.
