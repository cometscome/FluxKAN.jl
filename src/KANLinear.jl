function _weight_linear(weight, x)
    tail = size(x)[2:end]
    y = weight * reshape(x, size(x, 1), :)
    return reshape(y, size(weight, 1), tail...)
end

function _scaled_spline_weight(weight, scaler, out_dim, in_dim, n_basis)
    scaler === nothing && return weight
    weight_edges = reshape(weight, out_dim, n_basis, in_dim)
    scaler_edges = reshape(scaler, out_dim, 1, in_dim)
    return reshape(weight_edges .* scaler_edges, out_dim, n_basis * in_dim)
end

function _efficientkan_uniform(rng, out_dim, in_dim, scale, ::Type{T}) where {T}
    # Match torch.nn.init.kaiming_uniform_(a=sqrt(5) * scale), as used by
    # EfficientKAN for the residual weights and standalone spline scalers.
    a = sqrt(T(5)) * T(scale)
    gain = sqrt(T(2) / (one(T) + a^2))
    bound = sqrt(T(3)) * gain / sqrt(T(in_dim))
    return rand(rng, T, out_dim, in_dim) .* (2 * bound) .- bound
end

function _initial_spline_weight(
    rng,
    grid,
    in_dim,
    out_dim,
    grid_size,
    spline_order,
    scale_noise,
    ::Type{T},
) where {T}
    n_basis = grid_size + spline_order
    core_columns = (spline_order + 1):(size(grid, 2) - spline_order)
    core_input = grid[:, core_columns]
    basis = bspline_basis(core_input, grid, spline_order)
    noise = (rand(rng, T, grid_size + 1, in_dim, out_dim) .- T(0.5)) .*
            (T(scale_noise) / T(grid_size))
    weight = Matrix{T}(undef, out_dim, in_dim * n_basis)

    for feature in 1:in_dim
        rows = ((feature - 1) * n_basis + 1):(feature * n_basis)
        design = transpose(view(basis, rows, :))
        fitted = design \ view(noise, :, feature, :)
        view(weight, :, rows) .= transpose(fitted)
    end
    return weight
end

"""
    KANLinear(in_dim, out_dim; grid_size=5, spline_order=3,
              grid_min=-1, grid_max=1, base_activation=SiLU,
              scale_base=1, scale_spline=1, scale_noise=0.1,
              standalone_spline_scale=true, grid_eps=0.02, rng=default_rng())

Flux implementation of an EfficientKAN-style B-spline KAN layer. Every
input-output edge learns a linear combination of Cox-de Boor B-spline bases;
the edge spline functions are added to a SiLU residual/base branch.
"""
struct KANLinear{BW,SW,SS,G,A,T}
    base_weight::BW
    spline_weight::SW
    spline_scaler::SS
    grid::G
    base_activation::A
    in_dim::Int
    out_dim::Int
    grid_size::Int
    spline_order::Int
    grid_eps::T
end

function KANLinear(
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
    rng=Random.default_rng(),
)
    _validate_layer_dimensions(in_dim, out_dim)
    _validate_spline(grid_size, spline_order, grid_min, grid_max)
    0 <= grid_eps <= 1 || throw(ArgumentError(
        "grid_eps must lie in [0, 1], got $grid_eps",
    ))

    in_dim = Int(in_dim)
    out_dim = Int(out_dim)
    grid_size = Int(grid_size)
    spline_order = Int(spline_order)
    T = Float32
    grid = _extended_spline_grid(
        in_dim,
        grid_size,
        spline_order,
        grid_min,
        grid_max,
        T,
    )
    base_weight = _efficientkan_uniform(rng, out_dim, in_dim, scale_base, T)
    spline_weight = _initial_spline_weight(
        rng,
        grid,
        in_dim,
        out_dim,
        grid_size,
        spline_order,
        scale_noise,
        T,
    )
    spline_scaler = standalone_spline_scale ?
                    _efficientkan_uniform(rng, out_dim, in_dim, scale_spline, T) : nothing
    standalone_spline_scale || (spline_weight .*= T(scale_spline))
    return KANLinear(
        base_weight,
        spline_weight,
        spline_scaler,
        grid,
        base_activation,
        in_dim,
        out_dim,
        grid_size,
        spline_order,
        T(grid_eps),
    )
end

export KANLinear
Flux.@layer KANLinear trainable = (base_weight, spline_weight, spline_scaler)

function (m::KANLinear)(x)
    _check_input_dimension(x, m.in_dim)
    basis = bspline_basis(x, m.grid, m.spline_order)
    spline_weight = _scaled_spline_weight(
        m.spline_weight,
        m.spline_scaler,
        m.out_dim,
        m.in_dim,
        m.grid_size + m.spline_order,
    )
    base_output = _weight_linear(m.base_weight, m.base_activation.(x))
    return base_output .+ _weight_linear(spline_weight, basis)
end

function _adaptive_spline_grid(x, grid_size, spline_order, grid_eps, margin, ::Type{T}) where {T}
    margin > 0 || throw(ArgumentError("margin must be positive, got $margin"))
    in_dim, sample_count = size(x)
    sample_count >= grid_size + spline_order || throw(ArgumentError(
        "grid update needs at least $(grid_size + spline_order) samples, got $sample_count",
    ))
    new_grid = Matrix{T}(undef, in_dim, grid_size + 2 * spline_order + 1)
    quantile_indices = round.(Int, range(1, sample_count; length=grid_size + 1))

    for feature in 1:in_dim
        sorted_input = sort(collect(view(x, feature, :)))
        adaptive = sorted_input[quantile_indices]
        uniform_step = (adaptive[end] - adaptive[1] + 2 * margin) / grid_size
        uniform = adaptive[1] - margin .+ uniform_step .* (0:grid_size)
        # A purely adaptive grid collapses for a constant feature. In that
        # degenerate case there are no meaningful quantiles, so use the local
        # uniform grid irrespective of `grid_eps` and retain a valid basis.
        mixing = adaptive[1] == adaptive[end] ? one(grid_eps) : grid_eps
        core = mixing .* uniform .+ (1 - mixing) .* adaptive
        left = core[1] .- uniform_step .* (spline_order:-1:1)
        right = core[end] .+ uniform_step .* (1:spline_order)
        new_grid[feature, :] .= T.(vcat(left, core, right))
    end
    return new_grid
end

function _refit_spline_grid(
    x,
    old_grid,
    spline_weight,
    grid_size,
    spline_order,
    grid_eps,
    margin,
)
    x_matrix = reshape(x, size(x, 1), :)
    in_dim = size(x_matrix, 1)
    out_dim = size(spline_weight, 1)
    n_basis = grid_size + spline_order
    old_basis = bspline_basis(x_matrix, old_grid, spline_order)
    new_grid = _adaptive_spline_grid(
        x_matrix,
        grid_size,
        spline_order,
        grid_eps,
        margin,
        eltype(old_grid),
    )
    new_basis = bspline_basis(x_matrix, new_grid, spline_order)
    new_weight = similar(spline_weight)

    for feature in 1:in_dim
        rows = ((feature - 1) * n_basis + 1):(feature * n_basis)
        old_edge_basis = view(old_basis, rows, :)
        new_edge_basis = view(new_basis, rows, :)
        old_edge_weight = view(spline_weight, :, rows)
        edge_values = transpose(old_edge_basis) * transpose(old_edge_weight)
        fitted = transpose(new_edge_basis) \ edge_values
        view(new_weight, :, rows) .= transpose(fitted)
    end
    return new_grid, new_weight
end

"""
    update_grid!(layer::KANLinear, x; margin=0.01, grid_eps=layer.grid_eps)

Adapt each feature's knot grid to the sample distribution in `x`, then refit
the spline coefficients by least squares so the represented spline functions
are preserved on those samples. Grid updates are intended to run between
optimizer steps on CPU.
"""
function update_grid!(m::KANLinear, x; margin=0.01, grid_eps=m.grid_eps)
    _check_input_dimension(x, m.in_dim)
    0 <= grid_eps <= 1 || throw(ArgumentError(
        "grid_eps must lie in [0, 1], got $grid_eps",
    ))
    new_grid, new_weight = _refit_spline_grid(
        x,
        m.grid,
        m.spline_weight,
        m.grid_size,
        m.spline_order,
        grid_eps,
        margin,
    )
    copyto!(m.grid, new_grid)
    copyto!(m.spline_weight, new_weight)
    return m
end

export update_grid!

"""EfficientKAN-style coefficient magnitude and entropy regularization."""
function kan_regularization(
    m::KANLinear;
    regularize_activation=1,
    regularize_entropy=1,
)
    n_basis = m.grid_size + m.spline_order
    weights = reshape(abs.(m.spline_weight), m.out_dim, n_basis, m.in_dim)
    edge_magnitudes = dropdims(sum(weights; dims=2) ./ n_basis; dims=2)
    activation_penalty = sum(edge_magnitudes)
    probability = edge_magnitudes ./ (activation_penalty + eps(eltype(edge_magnitudes)))
    entropy_penalty = -sum(probability .* log.(probability .+ eps(eltype(probability))))
    return regularize_activation * activation_penalty + regularize_entropy * entropy_penalty
end

export kan_regularization
