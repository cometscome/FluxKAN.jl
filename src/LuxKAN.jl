function _lux_glorot_uniform(rng::AbstractRNG, out_dim, in_dim)
    bound = sqrt(Float32(6) / Float32(in_dim + out_dim))
    return rand(rng, Float32, out_dim, in_dim) .* (2f0 * bound) .- bound
end

function _lux_channel_view(v, x)
    return reshape(v, length(v), ntuple(_ -> 1, ndims(x) - 1)...)
end

function _lux_linear(weight, x)
    tail = size(x)[2:end]
    y = weight * reshape(x, size(x, 1), :)
    return reshape(y, size(weight, 1), tail...)
end

function _lux_linear(weight, bias, x)
    y = _lux_linear(weight, x)
    return y .+ _lux_channel_view(bias, y)
end

function _lux_layernorm(x, scale, bias)
    μ = sum(x; dims=1) ./ size(x, 1)
    centered = x .- μ
    variance = sum(abs2, centered; dims=1) ./ size(x, 1)
    normalized = centered ./ sqrt.(variance .+ convert(eltype(x), 1.0f-5))
    return normalized .* _lux_channel_view(scale, x) .+ _lux_channel_view(bias, x)
end

function _lux_norm_parameters(out_dim)
    out_dim == 1 && return NamedTuple()
    return (norm_scale=ones(Float32, out_dim), norm_bias=zeros(Float32, out_dim))
end

function _lux_input_norm_parameters(in_dim, use_layernorm)
    (!use_layernorm || in_dim == 1) && return NamedTuple()
    return (input_scale=ones(Float32, in_dim), input_bias=zeros(Float32, in_dim))
end

"""
    LuxKALnet(in_dim, out_dim; polynomial_order=3, base_activation=SiLU)

LuxCore implementation of `KALnet`. Use `Lux.setup(rng, layer)` to initialize
its explicit parameter and state NamedTuples.
"""
struct LuxKALnet{A} <: LuxCore.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    polynomial_order::Int
    base_activation::A
end

function LuxKALnet(in_dim, out_dim; polynomial_order=3, base_activation=SiLU)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_polynomial_order(polynomial_order)
    return LuxKALnet(Int(in_dim), Int(out_dim), Int(polynomial_order), base_activation)
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LuxKALnet)
    weights = (
        base_weight=_lux_glorot_uniform(rng, m.out_dim, m.in_dim),
        poly_weight=_lux_glorot_uniform(
            rng,
            m.out_dim,
            m.in_dim * (m.polynomial_order + 1),
        ),
    )
    return merge(weights, _lux_norm_parameters(m.out_dim))
end

LuxCore.initialstates(::AbstractRNG, ::LuxKALnet) = NamedTuple()

function (m::LuxKALnet)(x, ps, st)
    _check_input_dimension(x, m.in_dim)
    base_output = _lux_linear(ps.base_weight, m.base_activation.(x))
    basis = _polynomial_basis(
        compute_legendre_polynomials(_polynomial_input(x), m.polynomial_order),
    )
    y = base_output .+ _lux_linear(ps.poly_weight, basis)
    m.out_dim > 1 && (y = _lux_layernorm(y, ps.norm_scale, ps.norm_bias))
    return m.base_activation.(y), st
end

export LuxKALnet

"""LuxCore implementation of the TorchKAN-style `KACnet` layer."""
struct LuxKACnet{A} <: LuxCore.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    polynomial_order::Int
    base_activation::A
end

function LuxKACnet(in_dim, out_dim; polynomial_order=3, base_activation=SiLU)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_polynomial_order(polynomial_order)
    return LuxKACnet(Int(in_dim), Int(out_dim), Int(polynomial_order), base_activation)
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LuxKACnet)
    weights = (
        base_weight=_lux_glorot_uniform(rng, m.out_dim, m.in_dim),
        poly_weight=_lux_glorot_uniform(
            rng,
            m.out_dim,
            m.in_dim * (m.polynomial_order + 1),
        ),
    )
    return merge(weights, _lux_norm_parameters(m.out_dim))
end

LuxCore.initialstates(::AbstractRNG, ::LuxKACnet) = NamedTuple()

function (m::LuxKACnet)(x, ps, st)
    _check_input_dimension(x, m.in_dim)
    base_output = _lux_linear(ps.base_weight, m.base_activation.(x))
    basis = _polynomial_basis(
        compute_chebyshev_polynomials(_polynomial_input(x), m.polynomial_order),
    )
    y = base_output .+ _lux_linear(ps.poly_weight, basis)
    m.out_dim > 1 && (y = _lux_layernorm(y, ps.norm_scale, ps.norm_bias))
    return m.base_activation.(y), st
end

export LuxKACnet

"""LuxCore implementation of the fixed-grid FastKAN-style `KAGnet` layer."""
struct LuxKAGnet{A,T} <: LuxCore.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    num_grids::Int
    base_activation::A
    grid_min::T
    grid_max::T
    use_layernorm::Bool
end

function LuxKAGnet(
    in_dim,
    out_dim;
    num_grids=8,
    base_activation=SiLU,
    grid_min=-2,
    grid_max=2,
    use_layernorm=true,
)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_grid(num_grids, grid_min, grid_max)
    return LuxKAGnet(
        Int(in_dim),
        Int(out_dim),
        Int(num_grids),
        base_activation,
        Float32(grid_min),
        Float32(grid_max),
        Bool(use_layernorm),
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LuxKAGnet)
    weights = (
        base_weight=_lux_glorot_uniform(rng, m.out_dim, m.in_dim),
        base_bias=zeros(Float32, m.out_dim),
        poly_weight=_lux_glorot_uniform(rng, m.out_dim, m.in_dim * m.num_grids),
    )
    return merge(weights, _lux_input_norm_parameters(m.in_dim, m.use_layernorm))
end

function LuxCore.initialstates(::AbstractRNG, m::LuxKAGnet)
    return (
        grids=_grid_values(m.num_grids, m.grid_min, m.grid_max, Float32),
        denominator=_grid_denominator(
            m.num_grids,
            m.grid_min,
            m.grid_max,
            Float32,
        ),
    )
end

function (m::LuxKAGnet)(x, ps, st)
    _check_input_dimension(x, m.in_dim)
    basis_input = if m.use_layernorm && m.in_dim > 1
        _lux_layernorm(x, ps.input_scale, ps.input_bias)
    else
        x
    end
    basis = gaussian_rbf_basis(basis_input, st.grids, st.denominator)
    base_output = _lux_linear(ps.base_weight, ps.base_bias, m.base_activation.(x))
    return base_output .+ _lux_linear(ps.poly_weight, basis), st
end

export LuxKAGnet

"""LuxCore implementation of the learnable-grid experimental `KAGLnet` layer."""
struct LuxKAGLnet{A,T} <: LuxCore.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    num_grids::Int
    base_activation::A
    grid_min::T
    grid_max::T
    hasbase::Bool
    use_layernorm::Bool
end

function LuxKAGLnet(
    in_dim,
    out_dim;
    num_grids=8,
    base_activation=SiLU,
    grid_min=-2,
    grid_max=2,
    hasbase=true,
    use_layernorm=true,
)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_grid(num_grids, grid_min, grid_max)
    return LuxKAGLnet(
        Int(in_dim),
        Int(out_dim),
        Int(num_grids),
        base_activation,
        Float32(grid_min),
        Float32(grid_max),
        Bool(hasbase),
        Bool(use_layernorm),
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LuxKAGLnet)
    weights = (
        poly_weight=_lux_glorot_uniform(rng, m.out_dim, m.in_dim * m.num_grids),
        grids=_grid_values(m.num_grids, m.grid_min, m.grid_max, Float32),
    )
    if m.hasbase
        weights = merge(
            (
                base_weight=_lux_glorot_uniform(rng, m.out_dim, m.in_dim),
                base_bias=zeros(Float32, m.out_dim),
            ),
            weights,
        )
    end
    return merge(weights, _lux_input_norm_parameters(m.in_dim, m.use_layernorm))
end

function LuxCore.initialstates(::AbstractRNG, m::LuxKAGLnet)
    return (
        denominator=_grid_denominator(
            m.num_grids,
            m.grid_min,
            m.grid_max,
            Float32,
        ),
    )
end

function (m::LuxKAGLnet)(x, ps, st)
    _check_input_dimension(x, m.in_dim)
    basis_input = if m.use_layernorm && m.in_dim > 1
        _lux_layernorm(x, ps.input_scale, ps.input_bias)
    else
        x
    end
    basis = gaussian_rbf_basis(basis_input, ps.grids, st.denominator)
    y = _lux_linear(ps.poly_weight, basis)
    if m.hasbase
        y = y .+ _lux_linear(ps.base_weight, ps.base_bias, m.base_activation.(x))
    end
    return y, st
end

export LuxKAGLnet

"""
    LuxKANLinear(in_dim, out_dim; grid_size=5, spline_order=3, ...)

LuxCore implementation of the EfficientKAN-style B-spline `KANLinear` layer.
The fixed knot grid is stored in state and all learned edge coefficients and
scales are explicit Lux parameters.
"""
struct LuxKANLinear{A,T} <: LuxCore.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    grid_size::Int
    spline_order::Int
    base_activation::A
    grid_min::T
    grid_max::T
    scale_base::T
    scale_spline::T
    scale_noise::T
    standalone_spline_scale::Bool
    grid_eps::T
end

function LuxKANLinear(
    in_dim,
    out_dim;
    grid_size=5,
    spline_order=3,
    grid_min=-1,
    grid_max=1,
    base_activation=SiLU,
    scale_base=1,
    scale_spline=1,
    scale_noise=0.1,
    standalone_spline_scale=true,
    grid_eps=0.02,
)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_spline(grid_size, spline_order, grid_min, grid_max)
    0 <= grid_eps <= 1 || throw(ArgumentError(
        "grid_eps must lie in [0, 1], got $grid_eps",
    ))
    return LuxKANLinear(
        Int(in_dim),
        Int(out_dim),
        Int(grid_size),
        Int(spline_order),
        base_activation,
        Float32(grid_min),
        Float32(grid_max),
        Float32(scale_base),
        Float32(scale_spline),
        Float32(scale_noise),
        Bool(standalone_spline_scale),
        Float32(grid_eps),
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, m::LuxKANLinear)
    grid = _extended_spline_grid(
        m.in_dim,
        m.grid_size,
        m.spline_order,
        m.grid_min,
        m.grid_max,
        Float32,
    )
    parameters = (
        base_weight=_efficientkan_uniform(
            rng,
            m.out_dim,
            m.in_dim,
            m.scale_base,
            Float32,
        ),
        spline_weight=_initial_spline_weight(
            rng,
            grid,
            m.in_dim,
            m.out_dim,
            m.grid_size,
            m.spline_order,
            m.scale_noise,
            Float32,
        ),
    )
    if m.standalone_spline_scale
        return merge(
            parameters,
            (
                spline_scaler=_efficientkan_uniform(
                    rng,
                    m.out_dim,
                    m.in_dim,
                    m.scale_spline,
                    Float32,
                ),
            ),
        )
    end
    return merge(
        parameters,
        (spline_weight=parameters.spline_weight .* m.scale_spline,),
    )
end

function LuxCore.initialstates(::AbstractRNG, m::LuxKANLinear)
    return (
        grid=_extended_spline_grid(
            m.in_dim,
            m.grid_size,
            m.spline_order,
            m.grid_min,
            m.grid_max,
            Float32,
        ),
    )
end

function (m::LuxKANLinear)(x, ps, st)
    _check_input_dimension(x, m.in_dim)
    basis = bspline_basis(x, st.grid, m.spline_order)
    scaler = m.standalone_spline_scale ? ps.spline_scaler : nothing
    spline_weight = _scaled_spline_weight(
        ps.spline_weight,
        scaler,
        m.out_dim,
        m.in_dim,
        m.grid_size + m.spline_order,
    )
    base_output = _lux_linear(ps.base_weight, m.base_activation.(x))
    return base_output .+ _lux_linear(spline_weight, basis), st
end

export LuxKANLinear

"""
    update_grid(layer::LuxKANLinear, x, parameters, state; margin=0.01,
                grid_eps=layer.grid_eps)

Return updated Lux parameters and state after adapting the knot grid and
refitting the B-spline coefficients on `x`.
"""
function update_grid(
    m::LuxKANLinear,
    x,
    ps,
    st;
    margin=0.01,
    grid_eps=m.grid_eps,
)
    _check_input_dimension(x, m.in_dim)
    0 <= grid_eps <= 1 || throw(ArgumentError(
        "grid_eps must lie in [0, 1], got $grid_eps",
    ))
    new_grid, new_weight = _refit_spline_grid(
        x,
        st.grid,
        ps.spline_weight,
        m.grid_size,
        m.spline_order,
        grid_eps,
        margin,
    )
    return merge(ps, (spline_weight=new_weight,)), merge(st, (grid=new_grid,))
end

export update_grid

function kan_regularization(
    m::LuxKANLinear,
    ps;
    regularize_activation=1,
    regularize_entropy=1,
)
    n_basis = m.grid_size + m.spline_order
    weights = reshape(abs.(ps.spline_weight), m.out_dim, n_basis, m.in_dim)
    edge_magnitudes = dropdims(sum(weights; dims=2) ./ n_basis; dims=2)
    activation_penalty = sum(edge_magnitudes)
    probability = edge_magnitudes ./ (activation_penalty + eps(eltype(edge_magnitudes)))
    entropy_penalty = -sum(probability .* log.(probability .+ eps(eltype(probability))))
    return regularize_activation * activation_penalty + regularize_entropy * entropy_penalty
end
