using FluxKAN
using Flux
using BenchmarkTools
function main()
    model_FluxKAN = Chain(KAGnet(6, 10, num_grids=5),
        KAGnet(10, 10, num_grids=5),
        KAGnet(10, 2, num_grids=5))
    display(model_FluxKAN)

    x = ones(Float32, 6)
    @benchmark $model_FluxKAN($x)

    model = Chain(Dense(6, 30),
        Dense(30, 30),
        Dense(30, 2))

    model2 = Chain(Dense(6, 30, relu), LayerNorm(30),
        Dense(30, 30, relu), LayerNorm(30),
        Dense(30, 2), LayerNorm(2))

end
main()