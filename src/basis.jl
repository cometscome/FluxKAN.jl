"""SiLU activation used by the residual/base branch of the KAN layers."""
SiLU(x) = x / (one(x) + exp(-x))

function _validate_layer_dimensions(in_dim, out_dim)
    in_dim isa Integer && in_dim > 0 ||
        throw(ArgumentError("in_dim must be a positive integer, got $in_dim"))
    out_dim isa Integer && out_dim > 0 ||
        throw(ArgumentError("out_dim must be a positive integer, got $out_dim"))
    return nothing
end

function _validate_polynomial_order(order)
    order isa Integer && order >= 0 ||
        throw(ArgumentError("polynomial_order must be a non-negative integer, got $order"))
    return nothing
end

function _validate_grid(num_grids, grid_min, grid_max)
    num_grids isa Integer && num_grids >= 2 ||
        throw(ArgumentError("num_grids must be an integer greater than one, got $num_grids"))
    grid_min < grid_max ||
        throw(ArgumentError("grid_min must be smaller than grid_max, got ($grid_min, $grid_max)"))
    return nothing
end

function _validate_spline(grid_size, spline_order, grid_min, grid_max)
    grid_size isa Integer && grid_size >= 1 || throw(ArgumentError(
        "grid_size must be a positive integer, got $grid_size",
    ))
    spline_order isa Integer && spline_order >= 0 || throw(ArgumentError(
        "spline_order must be a non-negative integer, got $spline_order",
    ))
    grid_min < grid_max || throw(ArgumentError(
        "grid_min must be smaller than grid_max, got ($grid_min, $grid_max)",
    ))
    return nothing
end

function _check_input_dimension(x, in_dim)
    ndims(x) >= 1 || throw(DimensionMismatch("KAN layers require an array input"))
    size(x, 1) == in_dim || throw(DimensionMismatch(
        "layer expects size(input, 1) == $in_dim, got input with size $(size(x))",
    ))
    return nothing
end

function _ones_like(x)
    y = zero(x)
    fill!(y, one(eltype(x)))
    return y
end

"""
    compute_legendre_polynomials(x, order)

Return `P₀(x), …, P_order(x)` using the Legendre three-term recurrence.
`x` should normally already lie in `[-1, 1]`.
"""
function compute_legendre_polynomials(x, order)
    _validate_polynomial_order(order)
    P0 = _ones_like(x)
    order == 0 && return [P0]

    polynomials = [P0, copy(x)]
    for n in 1:(order - 1)
        Pn = ((2n + 1) .* x .* polynomials[end] .- n .* polynomials[end - 1]) ./ (n + 1)
        push!(polynomials, Pn)
    end
    return polynomials
end

export compute_legendre_polynomials

function ChainRulesCore.rrule(::typeof(compute_legendre_polynomials), x, order)
    y = compute_legendre_polynomials(x, order)

    derivatives = Vector{typeof(x)}(undef, order + 1)
    derivatives[1] = zero(x)
    if order >= 1
        derivatives[2] = _ones_like(x)
        for n in 1:(order - 1)
            derivatives[n + 2] = (
                (2n + 1) .* (y[n + 1] .+ x .* derivatives[n + 1]) .-
                n .* derivatives[n]
            ) ./ (n + 1)
        end
    end

    function pullback(ybar)
        ybar = unthunk(ybar)
        if ybar isa AbstractZero
            return NoTangent(), ZeroTangent(), NoTangent()
        end
        xbar = zero(x)
        for n in eachindex(derivatives)
            xbar .+= derivatives[n] .* unthunk(ybar[n])
        end
        return NoTangent(), xbar, NoTangent()
    end

    return y, pullback
end

"""
    compute_chebyshev_polynomials(x, order)

Return first-kind Chebyshev polynomials `T₀(x), …, T_order(x)` using their
three-term recurrence. `x` should normally already lie in `[-1, 1]`.
"""
function compute_chebyshev_polynomials(x, order)
    _validate_polynomial_order(order)
    T0 = _ones_like(x)
    order == 0 && return [T0]

    polynomials = [T0, copy(x)]
    for _ in 1:(order - 1)
        push!(polynomials, 2 .* x .* polynomials[end] .- polynomials[end - 1])
    end
    return polynomials
end

export compute_chebyshev_polynomials

function ChainRulesCore.rrule(::typeof(compute_chebyshev_polynomials), x, order)
    y = compute_chebyshev_polynomials(x, order)

    derivatives = Vector{typeof(x)}(undef, order + 1)
    derivatives[1] = zero(x)
    if order >= 1
        derivatives[2] = _ones_like(x)
        for n in 1:(order - 1)
            derivatives[n + 2] =
                2 .* y[n + 1] .+ 2 .* x .* derivatives[n + 1] .- derivatives[n]
        end
    end

    function pullback(ybar)
        ybar = unthunk(ybar)
        if ybar isa AbstractZero
            return NoTangent(), ZeroTangent(), NoTangent()
        end
        xbar = zero(x)
        for n in eachindex(derivatives)
            xbar .+= derivatives[n] .* unthunk(ybar[n])
        end
        return NoTangent(), xbar, NoTangent()
    end

    return y, pullback
end

_polynomial_basis(polynomials::AbstractVector{<:AbstractArray}) = reduce(vcat, polynomials)

"""Map polynomial inputs pointwise to the stable interval `(-1, 1)`."""
_polynomial_input(x) = tanh.(x)

function _grid_values(num_grids, grid_min, grid_max, ::Type{T}=Float32) where {T}
    return collect(range(T(grid_min), T(grid_max); length=num_grids))
end

function _grid_denominator(num_grids, grid_min, grid_max, ::Type{T}=Float32) where {T}
    return (T(grid_max) - T(grid_min)) / T(num_grids - 1)
end

"""
    gaussian_rbf_basis(x, grids, denominator)

Evaluate scalar Gaussian radial basis functions for every input feature. The
first dimension of the result is ordered by feature, then grid center, and has
length `size(x, 1) * length(grids)`.
"""
function gaussian_rbf_basis(x, grids, denominator)
    _check_input_dimension(x, size(x, 1))
    isempty(grids) && throw(ArgumentError("grids must contain at least one center"))
    iszero(denominator) && throw(ArgumentError("denominator must be nonzero"))
    tail = size(x)[2:end]
    xshape = (1, size(x, 1), tail...)
    gridshape = (length(grids), 1, ntuple(_ -> 1, length(tail))...)
    values = exp.(-((reshape(x, xshape) .- reshape(grids, gridshape)) ./ denominator) .^ 2)
    return reshape(values, size(x, 1) * length(grids), tail...)
end

export gaussian_rbf_basis

function _extended_spline_grid(
    in_dim,
    grid_size,
    spline_order,
    grid_min,
    grid_max,
    ::Type{T}=Float32,
) where {T}
    step = (T(grid_max) - T(grid_min)) / T(grid_size)
    knots = collect(
        range(
            T(grid_min) - T(spline_order) * step,
            T(grid_max) + T(spline_order) * step;
            length=grid_size + 2 * spline_order + 1,
        ),
    )
    return repeat(reshape(knots, 1, :), in_dim, 1)
end

function _safe_spline_ratio(numerator, denominator)
    zero_denominator = denominator .== zero(eltype(denominator))
    safe_denominator = ifelse.(
        zero_denominator,
        one(eltype(denominator)),
        denominator,
    )
    return ifelse.(
        zero_denominator,
        zero(eltype(numerator)),
        numerator ./ safe_denominator,
    )
end

"""
    bspline_basis(x, grid, spline_order)

Evaluate Cox-de Boor B-spline basis functions for feature-first input `x`.
`grid` can be a shared knot vector or an `in_dim × n_knots` matrix. The result
has `(n_knots - spline_order - 1) * in_dim` rows, ordered by basis function
within each input feature.
"""
function bspline_basis(x, grid, spline_order)
    _validate_polynomial_order(spline_order)
    _check_input_dimension(x, size(x, 1))

    in_dim = size(x, 1)
    n_knots = grid isa AbstractVector ? length(grid) : size(grid, 2)
    grid_features = grid isa AbstractVector ? 1 : size(grid, 1)
    grid_features in (1, in_dim) || throw(DimensionMismatch(
        "grid must have one row or $in_dim rows, got size $(size(grid))",
    ))
    n_knots >= spline_order + 2 || throw(ArgumentError(
        "spline_order=$spline_order requires at least $(spline_order + 2) knots",
    ))

    tail = size(x)[2:end]
    singleton_tail = ntuple(_ -> 1, length(tail))
    x_view = reshape(x, 1, in_dim, tail...)
    knot_matrix =
        grid isa AbstractVector ? reshape(grid, :, 1) : PermutedDimsArray(grid, (2, 1))
    knots = reshape(knot_matrix, n_knots, grid_features, singleton_tail...)

    lower = selectdim(knots, 1, 1:(n_knots - 1))
    upper = selectdim(knots, 1, 2:n_knots)
    basis = ifelse.(
        (x_view .>= lower) .& (x_view .< upper),
        one(eltype(x)),
        zero(eltype(x)),
    )

    for degree in 1:spline_order
        next_count = n_knots - degree - 1
        left_low = selectdim(knots, 1, 1:next_count)
        left_high = selectdim(knots, 1, (degree + 1):(n_knots - 1))
        right_low = selectdim(knots, 1, 2:(n_knots - degree))
        right_high = selectdim(knots, 1, (degree + 2):n_knots)

        left = _safe_spline_ratio(x_view .- left_low, left_high .- left_low)
        right = _safe_spline_ratio(right_high .- x_view, right_high .- right_low)
        basis =
            left .* selectdim(basis, 1, 1:next_count) .+
            right .* selectdim(basis, 1, 2:(next_count + 1))
    end

    n_basis = n_knots - spline_order - 1
    return reshape(basis, n_basis * in_dim, tail...)
end

export bspline_basis
