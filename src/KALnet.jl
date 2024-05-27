

mutable struct KALnet{in_dim,out_dim,polynomial_order}
    base_weight
    poly_weight
    layer_norm
    base_activation
    in_dim::Int64
    out_dim::Int64
    polynomial_order::Int64
end

function KALnet(in_dim, out_dim; polynomial_order=3, base_activation=SiLU)
    base_weight = Dense(in_dim, out_dim; bias=false)
    poly_weight = Dense(in_dim * (polynomial_order + 1), out_dim; bias=false)
    if out_dim == 1
        layer_norm = Dense(out_dim, out_dim; bias=false)
    else
        layer_norm = LayerNorm(out_dim)
    end
    return KALnet{in_dim,out_dim,polynomial_order}(base_weight,
        poly_weight, layer_norm, base_activation, in_dim, out_dim, polynomial_order)
end
function KALnet(base_weight, poly_weight, layer_norm, base_activation, in_dim, out_dim, polynomial_order)
    return KALnet{in_dim,out_dim,polynomial_order}(base_weight, poly_weight,
        layer_norm, base_activation,
        in_dim, out_dim, polynomial_order
    )
end

export KALnet
Flux.@layer KALnet

SiLU(x) = x / (1 + exp(-x))

function compute_legendre_polynomials(x, order)
    # Base case polynomials P0 and P1
    P0 = ones(size(x)...)#x.new_ones(x.shape)  # P0 = 1 for all x
    if order == 0
        #return P0
        return [P0]
    end
    P1 = deepcopy(x)
    legendre_polys = [P0, P1]

    # Compute higher order polynomials using recurrence
    for n = 1:order-1
        Pn = ((2.0 * n + 1.0) .* x .* legendre_polys[end] - n .* legendre_polys[end-1]) ./ (n + 1.0)
        push!(legendre_polys, Pn)
    end
    return legendre_polys
end
export compute_legendre_polynomials

function ChainRulesCore.rrule(::typeof(compute_legendre_polynomials), x, order)
    # Base case polynomials P0 and P1
    P0 = ones(size(x)...)#x.new_ones(x.shape)  # P0 = 1 for all x
    if order == 0
        y = [P0]
    else
        P1 = deepcopy(x)
        legendre_polys = [P0, P1]
        # Compute higher order polynomials using recurrence
        for n = 1:order-1
            Pn = ((2.0 * n + 1.0) .* x .* legendre_polys[end] - n .* legendre_polys[end-1]) ./ (n + 1.0)
            push!(legendre_polys, Pn)
        end
        y = legendre_polys
    end

    function pullback(ybar)
        sbar = NoTangent()
        dP0 = zero(x)
        if order == 0
            #dlegendre_polys = [dP0]
            dlegendre_polys = dP0
        else
            dP1 = ones(size(x)...)
            dlegendre_polys = [dP0, dP1]
            for n = 1:order-1
                dPn = (n + 1) * legendre_polys[n+2] + x .* dlegendre_polys[end]
                # ((2.0 * n + 1.0) .* x .* legendre_polys[end] - n .* legendre_polys[end-1]) ./ (n + 1.0)
                push!(dlegendre_polys, dPn)
            end
        end
        dLdPdPdx = zero(x)
        for n = 1:length(ybar)
            dLdPdPdx .+= dlegendre_polys[n] .* ybar[n]
        end
        return sbar, dLdPdPdx, sbar
    end
    return y, pullback
end

function (m::KALnet{in_dim,out_dim,polynomial_order})(x) where {in_dim,out_dim,polynomial_order}
    y = KALnet_forward(x, m.base_weight, m.poly_weight, m.layer_norm, m.base_activation, polynomial_order)
end

function normalize_x(x, xmin, dx)
    return 2 * (x .- xmin) / dx .- 1
end

function KALnet_forward(x, base_weight, poly_weight, layer_norm, base_activation, polynomial_order)
    # Apply base activation to input and then linear transform with base weights
    base_output = base_weight(base_activation.(x))
    # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
    xmin = minimum(x)
    xmax = maximum(x)
    dx = xmax - xmin
    x_normalized = normalize_x(x, xmin, dx)
    # Compute Legendre polynomials for the normalized x
    legendre_polys = compute_legendre_polynomials(x_normalized, polynomial_order)
    legendre_basis = cat(legendre_polys..., dims=1)
    # Compute polynomial output using polynomial weights
    poly_output = poly_weight(legendre_basis)
    # Combine base and polynomial outputs, normalize, and activate
    y = base_activation.(layer_norm(base_output .+ poly_output))
    return y
end

