module FluxKAN
using Flux
using ChainRulesCore
using LinearAlgebra
using LuxCore
using Random

include("basis.jl")
include("./KANLinear.jl")
include("./KALnet.jl")
include("./KACnet.jl")
include("./KAGnet.jl")
include("./KAGLnet.jl")
include("./LuxKAN.jl")
include("./examples.jl")

end
