"""Learnable Gaussian radial basis grid used by `KAGLnet`."""
struct Radial_distribution_function_L{G,D,T}
    grids::G
    denominator::D
    num_grids::Int
    grid_max::T
    grid_min::T
end

function Radial_distribution_function_L(num_grids, grid_min, grid_max)
    _validate_grid(num_grids, grid_min, grid_max)
    T = Float32
    grids = _grid_values(num_grids, grid_min, grid_max, T)
    denominator = _grid_denominator(num_grids, grid_min, grid_max, T)
    return Radial_distribution_function_L(
        grids,
        denominator,
        Int(num_grids),
        T(grid_max),
        T(grid_min),
    )
end

Flux.@layer Radial_distribution_function_L trainable = (grids,)

(m::Radial_distribution_function_L)(x) = gaussian_rbf_basis(x, m.grids, m.denominator)

"""Compatibility wrapper for the original misspelled internal function."""
rdf_foward_L(x, num_grids, grids, denominator) =
    rdf_foward(x, num_grids, grids, denominator)

"""
    KAGLnet(in_dim, out_dim; num_grids=8, base_activation=SiLU,
            grid_min=-2, grid_max=2, hasbase=true, use_layernorm=true,
            rng=default_rng())

Experimental FastKAN-style layer whose Gaussian RBF centers are trainable.
The RBF width remains fixed at its initial grid spacing.
"""
struct KAGLnet{BW,PW,N,A,R}
    base_weight::BW
    poly_weight::PW
    input_norm::N
    base_activation::A
    in_dim::Int
    out_dim::Int
    num_grids::Int
    rdf::R
    hasbase::Bool
end

function KAGLnet(
    in_dim,
    out_dim;
    num_grids=8,
    base_activation=SiLU,
    grid_max=2,
    grid_min=-2,
    hasbase=true,
    use_layernorm=true,
    rng=Random.default_rng(),
)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_grid(num_grids, grid_min, grid_max)

    init = Flux.glorot_uniform(rng)
    base_weight = hasbase ? Dense(in_dim => out_dim; init) : nothing
    poly_weight = Dense(in_dim * num_grids => out_dim; bias=false, init)
    input_norm = use_layernorm && in_dim > 1 ? LayerNorm(in_dim) : identity
    rdf = Radial_distribution_function_L(num_grids, grid_min, grid_max)
    return KAGLnet(
        base_weight,
        poly_weight,
        input_norm,
        base_activation,
        Int(in_dim),
        Int(out_dim),
        Int(num_grids),
        rdf,
        Bool(hasbase),
    )
end

export KAGLnet
Flux.@layer KAGLnet trainable = (base_weight, poly_weight, input_norm, rdf)

function (m::KAGLnet)(x)
    _check_input_dimension(x, m.in_dim)
    rbf_output = m.poly_weight(m.rdf(m.input_norm(x)))
    m.hasbase || return rbf_output
    return m.base_weight(m.base_activation.(x)) .+ rbf_output
end
