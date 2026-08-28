using BenchmarkTools
using Flux
using FluxKAN
using Lux
using Random

flux_gradient(layer, x) = Flux.gradient(current -> sum(current(x)), layer)
lux_gradient(layer, x, ps, st) =
    Flux.gradient(current -> sum(first(Lux.apply(layer, x, current, st))), ps)

function make_suite(; in_dim=32, out_dim=64, batch_size=256)
    rng = Random.Xoshiro(1401)
    x = rand(rng, Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0
    dense = Flux.Dense(in_dim => out_dim)
    spline = KANLinear(in_dim, out_dim; rng=Random.Xoshiro(1402))
    gaussian = KAGnet(in_dim, out_dim; rng=Random.Xoshiro(1403))
    lux_spline = LuxKANLinear(in_dim, out_dim)
    lux_parameters, lux_state = Lux.setup(Random.Xoshiro(1404), lux_spline)

    suite = BenchmarkGroup()
    suite["forward"] = BenchmarkGroup()
    suite["gradient"] = BenchmarkGroup()
    suite["maintenance"] = BenchmarkGroup()

    suite["forward"]["Dense"] = @benchmarkable $dense($x)
    suite["forward"]["KANLinear"] = @benchmarkable $spline($x)
    suite["forward"]["KAGnet"] = @benchmarkable $gaussian($x)
    suite["forward"]["LuxKANLinear"] =
        @benchmarkable Lux.apply($lux_spline, $x, $lux_parameters, $lux_state)

    suite["gradient"]["KANLinear"] =
        @benchmarkable flux_gradient($spline, $x)
    suite["gradient"]["LuxKANLinear"] =
        @benchmarkable lux_gradient($lux_spline, $x, $lux_parameters, $lux_state)

    suite["maintenance"]["update_grid!"] = @benchmarkable update_grid!(candidate, $x) setup = (
        candidate=deepcopy($spline)
    )
    return suite
end
