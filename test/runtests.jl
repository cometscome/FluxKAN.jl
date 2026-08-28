using FluxKAN
using Flux
using LegendrePolynomials
using Lux
using MLDatasets
using Random
using Serialization
using Test

include("fixtures/efficientkan_reference.jl")

function finite_difference_gradient(f, x; h=1.0e-6)
    gradient = similar(x)
    for i in eachindex(x)
        plus = copy(x)
        minus = copy(x)
        plus[i] += h
        minus[i] -= h
        gradient[i] = (f(plus) - f(minus)) / (2h)
    end
    return gradient
end

tree_isfinite(x::Number) = isfinite(x)
tree_isfinite(x::AbstractArray) = all(isfinite, x)
tree_isfinite(x::NamedTuple) = all(tree_isfinite, values(x))
tree_isfinite(::Nothing) = true

function lux_parameters(m::KANLinear)
    parameters = (
        base_weight=copy(m.base_weight),
        spline_weight=copy(m.spline_weight),
    )
    m.spline_scaler === nothing && return parameters
    return merge(parameters, (spline_scaler=copy(m.spline_scaler),))
end

function lux_parameters(m::KALnet)
    return (
        base_weight=copy(m.base_weight.weight),
        poly_weight=copy(m.poly_weight.weight),
        norm_scale=copy(m.layer_norm.diag.scale),
        norm_bias=copy(m.layer_norm.diag.bias),
    )
end

function lux_parameters(m::KACnet)
    return (
        base_weight=copy(m.base_weight.weight),
        poly_weight=copy(m.poly_weight.weight),
        norm_scale=copy(m.layer_norm.diag.scale),
        norm_bias=copy(m.layer_norm.diag.bias),
    )
end

function lux_parameters(m::KAGnet)
    return (
        base_weight=copy(m.base_weight.weight),
        base_bias=copy(m.base_weight.bias),
        poly_weight=copy(m.poly_weight.weight),
        input_scale=copy(m.input_norm.diag.scale),
        input_bias=copy(m.input_norm.diag.bias),
    )
end

function lux_parameters(m::KAGLnet)
    return (
        base_weight=copy(m.base_weight.weight),
        base_bias=copy(m.base_weight.bias),
        poly_weight=copy(m.poly_weight.weight),
        grids=copy(m.rdf.grids),
        input_scale=copy(m.input_norm.diag.scale),
        input_bias=copy(m.input_norm.diag.bias),
    )
end

@testset "FluxKAN" begin
    @testset "optional MNIST extension" begin
        @test length(methods(mnist_kan)) == 1
        @test length(methods(MNIST_KAN)) == 1
    end

    @testset "basis values" begin
        x = range(-0.9, 0.9; length=12) |> collect
        for order in 0:5
            legendre = compute_legendre_polynomials(x, order)
            chebyshev = compute_chebyshev_polynomials(x, order)
            for n in 0:order
                @test legendre[n + 1] ≈ Pl.(x, n)
                @test chebyshev[n + 1] ≈ cos.(n .* acos.(x))
            end
        end

        gaussian_input = reshape(Float64[-1, 0.5], 1, :)
        gaussian_grid = Float64[-1, 0, 1]
        @test gaussian_rbf_basis(gaussian_input, gaussian_grid, 1.0) ≈ Float64[
            1 exp(-2.25)
            exp(-1) exp(-0.25)
            exp(-4) exp(-0.25)
        ] atol = 1.0e-15 rtol = 1.0e-15
    end

    @testset "basis gradients" begin
        x = Float64[-0.7, 0.2, 0.8]
        for basis_function in
            (compute_legendre_polynomials, compute_chebyshev_polynomials)
            for order in 0:5
                weights = [fill(Float64(n), size(x)) for n in 1:(order + 1)]
                objective(z) = sum(map(
                    (basis, weight) -> sum(basis .* weight),
                    basis_function(z, order),
                    weights,
                ))
                automatic = Flux.gradient(objective, x)[1]
                numerical = finite_difference_gradient(objective, x)
                @test automatic ≈ numerical atol = 1.0e-7 rtol = 1.0e-7
            end
        end

        grids = collect(range(-1.0, 1.0; length=5))
        grid_objective(z) = sum(gaussian_rbf_basis(x, z, 0.5))
        automatic = Flux.gradient(grid_objective, grids)[1]
        numerical = finite_difference_gradient(grid_objective, grids)
        @test automatic ≈ numerical atol = 1.0e-7 rtol = 1.0e-7
    end

    @testset "B-spline basis" begin
        @test bspline_basis(reshape([1.25], 1, 1), [0.0, 1.0, 2.0, 3.0], 1) ≈
              reshape([0.75, 0.25], 2, 1)

        # Cox-de Boor's degree-zero convention is half-open at the last knot.
        x = reshape(collect(range(-1.0, 0.99; length=31)), 1, :)
        grid_size = 7
        for spline_order in 0:4
            step = 2 / grid_size
            grid = collect(range(
                -1 - spline_order * step,
                1 + spline_order * step;
                length=grid_size + 2 * spline_order + 1,
            ))
            basis = bspline_basis(x, grid, spline_order)
            @test size(basis) == (grid_size + spline_order, size(x, 2))
            @test minimum(basis) >= 0
            @test vec(sum(basis; dims=1)) ≈ ones(size(x, 2)) atol = 1.0e-12
        end

        spline_order = 3
        grid_size = 8
        step = 2 / grid_size
        grid = collect(range(
            -1 - spline_order * step,
            1 + spline_order * step;
            length=grid_size + 2 * spline_order + 1,
        ))
        gradient_input = reshape(Float64[-0.73, -0.19, 0.37, 0.82], 1, :)
        weights = reshape(
            collect(range(-0.7, 1.1; length=(grid_size + spline_order) * 4)),
            grid_size + spline_order,
            4,
        )
        objective(z) = sum(bspline_basis(z, grid, spline_order) .* weights)
        automatic = Flux.gradient(objective, gradient_input)[1]
        numerical = finite_difference_gradient(objective, gradient_input)
        @test automatic ≈ numerical atol = 1.0e-6 rtol = 1.0e-6

        @test_throws ArgumentError bspline_basis(x, [-1.0, 0.0], 2)
        @test_throws DimensionMismatch bspline_basis(
            repeat(x, 2, 1),
            zeros(3, 8),
            2,
        )

        # The Cox-de Boor convention matches EfficientKAN exactly: knot
        # intervals are left-closed/right-open and values outside the knot
        # support are zero.
        order_zero_grid = Float64[-1, 0, 1]
        endpoint_input = reshape(Float64[-2, -1, 0, 1, 2], 1, :)
        @test bspline_basis(endpoint_input, order_zero_grid, 0) == Float64[
            0 1 0 0 0
            0 0 1 0 0
        ]
    end

    @testset "EfficientKAN independent reference fixture" begin
        fixture = EFFICIENTKAN_REFERENCE
        basis = bspline_basis(
            fixture.input,
            fixture.grid,
            fixture.spline_order,
        )
        @test basis ≈ fixture.basis atol = 2.0e-15 rtol = 2.0e-15

        layer = KANLinear(
            2,
            2;
            grid_size=fixture.grid_size,
            spline_order=fixture.spline_order,
        )
        copyto!(layer.grid, fixture.grid)
        copyto!(layer.base_weight, fixture.base_weight)
        copyto!(layer.spline_weight, fixture.spline_weight)
        copyto!(layer.spline_scaler, fixture.spline_scaler)
        @test layer(Float32.(fixture.input)) ≈ Float32.(fixture.output) atol = 2.0f-6 rtol = 2.0f-6
    end

    @testset "B-spline grid adaptation" begin
        rng = Random.Xoshiro(72)
        x = vcat(
            reshape(sort(rand(rng, Float32, 96) .^ 2 .* 1.8f0 .- 0.9f0), 1, :),
            reshape(sort(rand(rng, Float32, 96) .^ 3 .* 1.8f0 .- 0.9f0), 1, :),
        )

        flux_layer = KANLinear(2, 2; rng)
        old_grid = copy(flux_layer.grid)
        old_output = flux_layer(x)
        update_grid!(flux_layer, x)
        @test flux_layer.grid != old_grid
        @test maximum(abs, flux_layer(x) .- old_output) < 3.0f-3

        lux_layer = LuxKANLinear(2, 2)
        parameters, state = Lux.setup(rng, lux_layer)
        old_output = first(Lux.apply(lux_layer, x, parameters, state))
        new_parameters, new_state = update_grid(
            lux_layer,
            x,
            parameters,
            state,
        )
        new_output = first(Lux.apply(lux_layer, x, new_parameters, new_state))
        @test new_state.grid != state.grid
        @test maximum(abs, new_output .- old_output) < 2.0f-3

        constant_input = fill(0.25f0, 2, 96)
        constant_layer = KANLinear(2, 2; rng=Random.Xoshiro(73))
        constant_output = constant_layer(constant_input)
        update_grid!(constant_layer, constant_input; grid_eps=0)
        @test all(diff(constant_layer.grid; dims=2) .> 0)
        @test all(isfinite, constant_layer(constant_input))
        @test constant_layer(constant_input) ≈ constant_output atol = 2.0f-3 rtol = 2.0f-3
    end

    @testset "Flux layers" begin
        x = rand(Float32, 3, 4)
        constant_input = fill(0.5f0, 3, 4)
        sample = rand(Float32, 3)
        batch = hcat(sample, rand(Float32, 3, 3))

        for layer in (
            KANLinear(3, 2),
            KALnet(3, 2),
            KACnet(3, 2),
            KAGnet(3, 2),
            KAGLnet(3, 2),
        )
            output = @inferred layer(x)
            @test size(output) == (2, 4)
            @test all(isfinite, output)
            @test all(isfinite, layer(constant_input))
            @test layer(sample) ≈ layer(batch)[:, 1] atol = 2.0f-6 rtol = 2.0f-6

            input_gradient = Flux.gradient(z -> sum(layer(z)), x)[1]
            @test all(isfinite, input_gradient)
        end

        @test size(KALnet(3, 1)(x)) == (1, 4)
        @test size(KANLinear(3, 1)(x)) == (1, 4)
        @test size(KACnet(3, 1)(x)) == (1, 4)
        @test size(KAGnet(3, 1)(x)) == (1, 4)
        @test size(KAGLnet(3, 1)(x)) == (1, 4)
        @test size(KAGnet(3, 2; use_layernorm=false)(x)) == (2, 4)
        @test size(KAGLnet(3, 2; hasbase=false)(x)) == (2, 4)

        legacy_rbf = FluxKAN.rdf_foward(
            x,
            3,
            Float32[-1, 0, 1],
            1.0f0,
        )
        @test length(legacy_rbf) == 3
        @test all(size(basis) == size(x) for basis in legacy_rbf)

        @test_throws ArgumentError KALnet(0, 2)
        @test_throws ArgumentError KANLinear(2, 2; grid_size=0)
        @test_throws ArgumentError KANLinear(2, 2; spline_order=-1)
        @test_throws ArgumentError KANLinear(2, 2; grid_eps=1.1)
        @test_throws ArgumentError KACnet(2, 2; polynomial_order=-1)
        @test_throws ArgumentError KAGnet(2, 2; num_grids=1)
        @test_throws ArgumentError KAGLnet(2, 2; grid_min=1, grid_max=1)
        @test_throws ArgumentError gaussian_rbf_basis(x, Float32[], 1.0f0)
        @test_throws ArgumentError gaussian_rbf_basis(x, Float32[-1, 1], 0.0f0)
        @test_throws DimensionMismatch KALnet(3, 2)(rand(Float32, 4, 2))

        model = Flux.Chain(KALnet(3, 4), Flux.Dense(4 => 1))
        target = rand(Float32, 1, 4)
        optimizer_state = Flux.setup(Adam(), model)
        gradients = Flux.gradient(m -> Flux.mse(m(x), target), model)[1]
        Flux.update!(optimizer_state, model, gradients)
        @test all(isfinite, model(x))

        for constructor in (KANLinear, KALnet, KACnet, KAGnet, KAGLnet)
            first_layer = constructor(3, 2; rng=Random.Xoshiro(812))
            second_layer = constructor(3, 2; rng=Random.Xoshiro(812))
            @test first_layer(x) == second_layer(x)
        end
    end

    @testset "Lux layers" begin
        rng = Random.Xoshiro(123)
        x = rand(rng, Float32, 3, 4)
        sample = rand(rng, Float32, 3)
        batch = hcat(sample, rand(rng, Float32, 3, 3))

        layers = (
            LuxKANLinear(3, 2),
            LuxKALnet(3, 2),
            LuxKACnet(3, 2),
            LuxKAGnet(3, 2),
            LuxKAGLnet(3, 2),
        )
        for layer in layers
            parameters, state = Lux.setup(rng, layer)
            output, new_state = Lux.apply(layer, x, parameters, state)
            @test size(output) == (2, 4)
            @test all(isfinite, output)
            @test tree_isfinite(parameters)
            @test tree_isfinite(new_state)

            single_output = first(Lux.apply(layer, sample, parameters, state))
            batch_output = first(Lux.apply(layer, batch, parameters, state))[:, 1]
            @test single_output ≈ batch_output atol = 2.0f-6 rtol = 2.0f-6

            parameter_gradient = Flux.gradient(parameters) do ps
                sum(first(Lux.apply(layer, x, ps, state)))
            end[1]
            @test tree_isfinite(parameter_gradient)
        end

        for layer in (
            LuxKANLinear(3, 1),
            LuxKALnet(3, 1),
            LuxKACnet(3, 1),
            LuxKAGnet(3, 1; use_layernorm=false),
            LuxKAGLnet(3, 1; hasbase=false),
        )
            parameters, state = Lux.setup(rng, layer)
            output = first(Lux.apply(layer, x, parameters, state))
            @test size(output) == (1, 4)
            parameter_gradient = Flux.gradient(parameters) do ps
                sum(first(Lux.apply(layer, x, ps, state)))
            end[1]
            @test tree_isfinite(parameter_gradient)
        end
    end

    @testset "Flux and Lux numerical agreement" begin
        x = rand(Float32, 3, 5)

        flux_layers = (
            KANLinear(3, 2),
            KALnet(3, 2),
            KACnet(3, 2),
            KAGnet(3, 2),
            KAGLnet(3, 2),
        )
        lux_layers = (
            LuxKANLinear(3, 2),
            LuxKALnet(3, 2),
            LuxKACnet(3, 2),
            LuxKAGnet(3, 2),
            LuxKAGLnet(3, 2),
        )

        for (flux_layer, lux_layer) in zip(flux_layers, lux_layers)
            parameters = lux_parameters(flux_layer)
            state = if flux_layer isa KANLinear
                (grid=copy(flux_layer.grid),)
            elseif flux_layer isa KAGnet
                (grids=copy(flux_layer.rdf.grids), denominator=flux_layer.rdf.denominator)
            elseif flux_layer isa KAGLnet
                (denominator=flux_layer.rdf.denominator,)
            else
                NamedTuple()
            end
            lux_output = first(Lux.apply(lux_layer, x, parameters, state))
            @test flux_layer(x) ≈ lux_output atol = 2.0f-6 rtol = 2.0f-6
        end
    end

    @testset "B-spline learning" begin
        x = reshape(collect(range(-0.95f0, 0.95f0; length=128)), 1, :)
        target = sin.(3.0f0 * Float32(pi) .* x) .+ 0.25f0 .* x .^ 2

        flux_layer = KANLinear(
            1,
            1;
            grid_size=12,
            spline_order=3,
            standalone_spline_scale=false,
            rng=Random.Xoshiro(44),
        )
        initial_loss = Flux.mse(flux_layer(x), target)
        initial_spline_weight = copy(flux_layer.spline_weight)
        optimizer_state = Flux.setup(Adam(0.03), flux_layer)
        for _ in 1:300
            gradients = Flux.gradient(
                layer -> Flux.mse(layer(x), target),
                flux_layer,
            )[1]
            Flux.update!(optimizer_state, flux_layer, gradients)
        end
        final_loss = Flux.mse(flux_layer(x), target)
        @test final_loss < 1.0f-3
        @test final_loss < initial_loss / 100
        @test flux_layer.spline_weight != initial_spline_weight

        lux_layer = LuxKANLinear(
            1,
            1;
            grid_size=12,
            spline_order=3,
            standalone_spline_scale=false,
        )
        parameters, state = Lux.setup(Random.Xoshiro(45), lux_layer)
        loss(ps) = Flux.mse(
            first(Lux.apply(lux_layer, x, ps, state)),
            target,
        )
        initial_loss = loss(parameters)
        optimizer_state = Flux.setup(Adam(0.03), parameters)
        for _ in 1:300
            gradients = Flux.gradient(loss, parameters)[1]
            Flux.update!(optimizer_state, parameters, gradients)
        end
        final_loss = loss(parameters)
        @test final_loss < 1.0f-3
        @test final_loss < initial_loss / 100
    end

    @testset "grid update during optimization" begin
        rng = Random.Xoshiro(918)
        x = rand(rng, Float32, 2, 128) .* 1.8f0 .- 0.9f0
        target = reshape((x[1, :] .> x[2, :]), 1, :) .* 2.0f0 .- 1.0f0
        model = Flux.Chain(
            KANLinear(2, 6; grid_size=6, rng),
            KANLinear(6, 1; grid_size=6, rng),
        )
        optimizer_state = Flux.setup(Adam(0.02), model)
        loss(current_model) = Flux.mse(current_model(x), target)
        initial_loss = loss(model)
        for step in 1:120
            gradients = Flux.gradient(loss, model)[1]
            Flux.update!(optimizer_state, model, gradients)
            if step == 40
                update_grid!(model[1], x)
                update_grid!(model[2], model[1](x))
            end
        end
        @test isfinite(loss(model))
        @test loss(model) < initial_loss / 3
    end

    @testset "serialization and reproducibility" begin
        rng = Random.Xoshiro(921)
        x = rand(rng, Float32, 2, 7)

        flux_layer = KANLinear(2, 3; rng=Random.Xoshiro(922))
        flux_output = flux_layer(x)
        mktemp() do path, io
            serialize(io, flux_layer)
            close(io)
            restored = open(deserialize, path)
            @test restored(x) == flux_output
        end

        lux_layer = LuxKANLinear(2, 3)
        parameters, state = Lux.setup(Random.Xoshiro(923), lux_layer)
        lux_output = first(Lux.apply(lux_layer, x, parameters, state))
        mktemp() do path, io
            serialize(io, (lux_layer, parameters, state))
            close(io)
            restored_layer, restored_parameters, restored_state = open(deserialize, path)
            restored_output = first(Lux.apply(
                restored_layer,
                x,
                restored_parameters,
                restored_state,
            ))
            @test restored_output == lux_output
        end

        first_parameters, first_state = Lux.setup(
            Random.Xoshiro(924),
            LuxKANLinear(2, 3),
        )
        second_parameters, second_state = Lux.setup(
            Random.Xoshiro(924),
            LuxKANLinear(2, 3),
        )
        @test first_parameters == second_parameters
        @test first_state == second_state
    end
end
