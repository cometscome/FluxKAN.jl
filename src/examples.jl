"""
    mnist_kan(; kwargs...)

Train one of the Flux KAN variants on MNIST. Load `MLDatasets` before calling
this function; its package extension provides the implementation without making
the dataset stack a mandatory dependency of FluxKAN.
"""
function mnist_kan end

export mnist_kan

"""Deprecated compatibility alias for [`mnist_kan`](@ref)."""
function MNIST_KAN(; kwargs...)
    Base.depwarn("MNIST_KAN is deprecated; use mnist_kan instead", :MNIST_KAN)
    return mnist_kan(; kwargs...)
end

export MNIST_KAN
