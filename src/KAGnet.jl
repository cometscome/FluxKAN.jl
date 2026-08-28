"""
    Radial_distribution_function(num_grids, grid_min, grid_max)

Fixed Gaussian radial basis grid used by `KAGnet`. Calling the object returns a
feature-major basis matrix with `size(x, 1) * num_grids` rows.
"""
struct Radial_distribution_function{G,D,T}
    grids::G
    denominator::D
    num_grids::Int
    grid_max::T
    grid_min::T
end

function Radial_distribution_function(num_grids, grid_min, grid_max)
    _validate_grid(num_grids, grid_min, grid_max)
    T = Float32
    grids = _grid_values(num_grids, grid_min, grid_max, T)
    denominator = _grid_denominator(num_grids, grid_min, grid_max, T)
    return Radial_distribution_function(
        grids,
        denominator,
        Int(num_grids),
        T(grid_max),
        T(grid_min),
    )
end

Flux.@layer Radial_distribution_function trainable = ()

(m::Radial_distribution_function)(x) = gaussian_rbf_basis(x, m.grids, m.denominator)

"""Compatibility wrapper for the original misspelled internal function."""
rdf_foward(x, num_grids, grids, denominator) = begin
    length(grids) == num_grids || throw(DimensionMismatch(
        "num_grids=$num_grids but $(length(grids)) grid centers were supplied",
    ))
    [exp.(-((x .- grid) ./ denominator) .^ 2) for grid in grids]
end

"""
    KAGnet(in_dim, out_dim; num_grids=8, base_activation=SiLU,
           grid_min=-2, grid_max=2, use_layernorm=true, rng=default_rng())

Flux implementation of a FastKAN layer with fixed Gaussian RBF centers. In
accordance with FastKAN, LayerNorm is applied to the input of the RBF branch,
while the base branch operates on the unnormalized input. The two linear
outputs are added without an extra output activation.
"""
struct KAGnet{BW,PW,N,A,R}
    base_weight::BW
    poly_weight::PW
    input_norm::N
    base_activation::A
    in_dim::Int
    out_dim::Int
    num_grids::Int
    rdf::R
end

function KAGnet(
    in_dim,
    out_dim;
    num_grids=8,
    base_activation=SiLU,
    grid_max=2,
    grid_min=-2,
    use_layernorm=true,
    rng=Random.default_rng(),
)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_grid(num_grids, grid_min, grid_max)

    init = Flux.glorot_uniform(rng)
    base_weight = Dense(in_dim => out_dim; init)
    poly_weight = Dense(in_dim * num_grids => out_dim; bias=false, init)
    input_norm = use_layernorm && in_dim > 1 ? LayerNorm(in_dim) : identity
    rdf = Radial_distribution_function(num_grids, grid_min, grid_max)
    return KAGnet(
        base_weight,
        poly_weight,
        input_norm,
        base_activation,
        Int(in_dim),
        Int(out_dim),
        Int(num_grids),
        rdf,
    )
end

export KAGnet
Flux.@layer KAGnet trainable = (base_weight, poly_weight, input_norm)

function (m::KAGnet)(x)
    _check_input_dimension(x, m.in_dim)
    base_output = m.base_weight(m.base_activation.(x))
    rbf_output = m.poly_weight(m.rdf(m.input_norm(x)))
    return base_output .+ rbf_output
end
