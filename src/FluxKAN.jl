module FluxKAN
using Flux
using LinearAlgebra
using ChainRulesCore
# Write your package code here.

include("./KALnet.jl")
include("./KACnet.jl")
include("./KAGnet.jl")
include("./KAGLnet.jl")
include("./examples.jl")

end
