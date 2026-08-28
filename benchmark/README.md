# Benchmarks

Run the benchmark suite from the repository root:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/runbenchmarks.jl benchmark-v0.9.json
```

The suite measures Flux `Dense`, `KANLinear`, `KAGnet`, Lux forward passes,
Flux/Lux gradients, and adaptive-grid refitting with the same input shape.
Benchmark JSON files are intentionally not committed because timings depend on
the Julia version, BLAS, CPU, and thread count. Compare releases on the same
machine with the same `JULIA_NUM_THREADS` value.
