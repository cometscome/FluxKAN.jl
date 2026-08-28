"""
    KACnet(in_dim, out_dim; polynomial_order=3, base_activation=SiLU,
           rng=default_rng())

Flux implementation of the TorchKAN-style Kolmogorov-Arnold Chebyshev layer.
This is the KAC-Net residual architecture, not the base-branch-free ChebyKAN
layer. Inputs to the polynomial branch are transformed pointwise with `tanh`.
"""
struct KACnet{BW,PW,LN,A}
    base_weight::BW
    poly_weight::PW
    layer_norm::LN
    base_activation::A
    in_dim::Int
    out_dim::Int
    polynomial_order::Int
end

function KACnet(
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
    return KACnet(
        base_weight,
        poly_weight,
        layer_norm,
        base_activation,
        Int(in_dim),
        Int(out_dim),
        Int(polynomial_order),
    )
end

export KACnet
Flux.@layer KACnet trainable = (base_weight, poly_weight, layer_norm)

function (m::KACnet)(x)
    _check_input_dimension(x, m.in_dim)
    base_output = m.base_weight(m.base_activation.(x))
    chebyshev_basis = _polynomial_basis(
        compute_chebyshev_polynomials(_polynomial_input(x), m.polynomial_order),
    )
    poly_output = m.poly_weight(chebyshev_basis)
    return m.base_activation.(m.layer_norm(base_output .+ poly_output))
end
