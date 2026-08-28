using BenchmarkTools

include("benchmarks.jl")

suite = make_suite()
BenchmarkTools.tune!(suite)
results = BenchmarkTools.run(suite; verbose=true)
show(stdout, MIME("text/plain"), results)
println()

if !isempty(ARGS)
    BenchmarkTools.save(first(ARGS), results)
    println("Saved benchmark results to $(first(ARGS))")
end
