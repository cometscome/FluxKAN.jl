"""
    KALnet(in_dim, out_dim; polynomial_order=3, base_activation=SiLU,
           rng=default_rng())

Flux implementation of the TorchKAN-style Kolmogorov-Arnold Legendre layer.
The layer adds a residual base branch to a learned linear combination of
Legendre basis functions. Inputs to the polynomial branch are transformed with
`tanh` pointwise, avoiding batch-dependent min/max normalization.
"""
struct KALnet{BW,PW,LN,A}
    base_weight::BW
    poly_weight::PW
    layer_norm::LN
    base_activation::A
    in_dim::Int
    out_dim::Int
    polynomial_order::Int
end

function KALnet(
    in_dim,
    out_dim;
    polynomial_order=3,
    base_activation=SiLU,
    rng=Random.default_rng(),
)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_polynomial_order(polynomial_order)

    init = Flux.glorot_uniform(rng)
    base_weight = Dense(in_dim => out_dim; bias=false, init)
    poly_weight = Dense(
        in_dim * (polynomial_order + 1) => out_dim;
        bias=false,
        init,
    )
    layer_norm = out_dim == 1 ? identity : LayerNorm(out_dim)
    return KALnet(
        base_weight,
        poly_weight,
        layer_norm,
        base_activation,
        Int(in_dim),
        Int(out_dim),
        Int(polynomial_order),
    )
end

export KALnet
Flux.@layer KALnet trainable = (base_weight, poly_weight, layer_norm)

function (m::KALnet)(x)
    _check_input_dimension(x, m.in_dim)
    base_output = m.base_weight(m.base_activation.(x))
    legendre_basis = _polynomial_basis(
        compute_legendre_polynomials(_polynomial_input(x), m.polynomial_order),
    )
    poly_output = m.poly_weight(legendre_basis)
    return m.base_activation.(m.layer_norm(base_output .+ poly_output))
end
