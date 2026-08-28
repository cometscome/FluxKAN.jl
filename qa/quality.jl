using Aqua
using FluxKAN
using JET
using Random
using Test

@testset "Aqua" begin
    Aqua.test_all(FluxKAN)
end

@testset "JET public workloads" begin
    x = rand(Random.Xoshiro(1101), Float32, 3, 8)
    spline_layer = KANLinear(3, 2; rng=Random.Xoshiro(1102))
    polynomial_layer = KALnet(3, 2; rng=Random.Xoshiro(1103))
    gaussian_layer = KAGnet(3, 2; rng=Random.Xoshiro(1104))

    JET.@test_opt target_modules=(FluxKAN,) bspline_basis(
        x,
        spline_layer.grid,
        spline_layer.spline_order,
    )
    JET.@test_opt target_modules=(FluxKAN,) spline_layer(x)
    JET.@test_opt target_modules=(FluxKAN,) polynomial_layer(x)
    JET.@test_opt target_modules=(FluxKAN,) gaussian_layer(x)
end
