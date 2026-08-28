using FluxKAN
using Random

x = rand(Random.Xoshiro(1201), Float32, 2, 4)
@assert size(KANLinear(2, 3; rng=Random.Xoshiro(1202))(x)) == (3, 4)
@assert LuxKANLinear(2, 3) isa LuxKANLinear
