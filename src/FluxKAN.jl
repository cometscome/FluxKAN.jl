module FluxKAN
using Flux
using LinearAlgebra
using ChainRulesCore
# Write your package code here.

include("./KALnet.jl")
include("./KACnet.jl")
include("./examples.jl")

end
