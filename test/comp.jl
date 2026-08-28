using FluxKAN
using Flux
using BenchmarkTools

function main()
    kan_model = Chain(
        KAGnet(6, 10; num_grids=5),
        KAGnet(10, 10; num_grids=5),
        KAGnet(10, 2; num_grids=5),
    )
    display(kan_model)

    x = ones(Float32, 6)
    display(@benchmark $kan_model($x))

    dense_model = Chain(
        Dense(6 => 30),
        Dense(30 => 30),
        Dense(30 => 2),
    )
    display(@benchmark $dense_model($x))

    normalized_model = Chain(
        Dense(6 => 30, relu),
        LayerNorm(30),
        Dense(30 => 30, relu),
        LayerNorm(30),
        Dense(30 => 2),
        LayerNorm(2),
    )
    display(@benchmark $normalized_model($x))
end

main()
