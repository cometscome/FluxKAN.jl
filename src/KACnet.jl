
```
KACnet: 
Chebyshev polynomials version
```
mutable struct KACnet{in_dim,out_dim,polynomial_order}
    base_weight
    poly_weight
    layer_norm
    base_activation
    in_dim
    out_dim
    polynomial_order
end

function KACnet(in_dim, out_dim; polynomial_order=3, base_activation=SiLU)
    base_weight = Dense(in_dim, out_dim; bias=false)
    poly_weight = Dense(in_dim * (polynomial_order + 1), out_dim; bias=false)
    if out_dim == 1
        layer_norm = Dense(out_dim, out_dim; bias=false)
    else
        layer_norm = LayerNorm(out_dim)
    end
    return KACnet{in_dim,out_dim,polynomial_order}(base_weight,
        poly_weight, layer_norm, base_activation, in_dim, out_dim, polynomial_order)
end
function KACnet(base_weight, poly_weight, layer_norm, base_activation, in_dim, out_dim, polynomial_order)
    return KACnet{in_dim,out_dim,polynomial_order}(base_weight, poly_weight,
        layer_norm, base_activation,
        in_dim, out_dim, polynomial_order
    )
end

export KACnet
Flux.@layer KACnet

function compute_chebyshev_polynomials(x, order)
    # Base case polynomials P0 and P1
    P0 = zero(x)
    fill!(P0,1)
    #P0 = ones(eltype(x), size(x)...)#x.new_ones(x.shape)  # P0 = 1 for all x
    if order == 0
        #return P0
        return [P0]
    end
    P1 = deepcopy(x)
    chebyshev_polys = [P0, P1]

    # Compute higher order polynomials using recurrence
    for n = 1:order-1
        Cp1 = chebyshev_polys[end]
        Cp0 = chebyshev_polys[end-1]
        
        Pn = map((x,cp1,cp0) -> 2*x*cp1 - cp0,x,Cp1,Cp0)
        #Pn = 2 .* x .* chebyshev_polys[end] .- chebyshev_polys[end-1] #2x Tn - T_{n-1}
        #Pn = ((2.0 * n + 1.0) .* x .* chebyshev_polys[end] - n .* chebyshev_polys[end-1]) ./ (n + 1.0)
        push!(chebyshev_polys, Pn)
    end
    return chebyshev_polys
end
export compute_chebyshev_polynomials

function ChainRulesCore.rrule(::typeof(compute_chebyshev_polynomials), x, order)
    # Base case polynomials P0 and P1
    P0 = zero(x)
    fill!(P0,1)
    #P0 = ones(eltype(x), size(x)...)#x.new_ones(x.shape)  # P0 = 1 for all x
    if order == 0
        y = [P0]
    else
        P1 = deepcopy(x)
        chebyshev_polys = [P0, P1]
        # Compute higher order polynomials using recurrence
        for n = 1:order-1
            Cp1 = chebyshev_polys[end]
            Cp0 = chebyshev_polys[end-1]
        
            Pn = map((x,cp1,cp0) -> 2*x*cp1 - cp0,x,Cp1,Cp0)

            #Pn = 2 .* x .* chebyshev_polys[end] .- chebyshev_polys[end-1] #2x Tn - T_{n-1}
            #Pn = ((2.0 * n + 1.0) .* x .* chebyshev_polys[end] - n .* chebyshev_polys[end-1]) ./ (n + 1.0)
            push!(chebyshev_polys, Pn)
        end
        y = chebyshev_polys
    end

    function pullback(ybar)
        sbar = NoTangent()
        dP0 = zero(x)
        if order == 0
            #dchebyshev_polys = [dP0]
            dchebyshev_polys = dP0
        else
            
            dP1 = zero(x)
            fill!(dP1,1)
            #ones(eltype(x), size(x)...)
            dchebyshev_polys = [dP0, dP1]
            for n = 1:order-1
                dCp1 = dchebyshev_polys[end]
                dCp0 = dchebyshev_polys[end-1]
                dPn = map((x,cp1,cp0) -> 2*x*cp1 - cp0,x,dCp1,dCp0)
                #dPn = 2 .* x .* dchebyshev_polys[end] .- dchebyshev_polys[end-1] #2x Tn - T_{n-1}
                push!(dchebyshev_polys, dPn)
            end
        end
        dLdPdPdx = zero(x)
        for n = 1:length(ybar)
            dLdPdPdx .+= dchebyshev_polys[n] .* ybar[n] * (n - 1)
        end
        return sbar, dLdPdPdx, sbar
    end
    return y, pullback
end

function (m::KACnet{in_dim,out_dim,polynomial_order})(x) where {in_dim,out_dim,polynomial_order}
    y = KACnet_forward(x, m.base_weight, m.poly_weight, m.layer_norm, m.base_activation, polynomial_order)
end


function KACnet_forward(x, base_weight, poly_weight, layer_norm, base_activation, polynomial_order)
    # Apply base activation to input and then linear transform with base weights
    base_output = base_weight(map(x -> base_activation(x),x))#base_weight(base_activation.(x))
    # Normalize x to the range [-1, 1] for stable chebyshev polynomial computation
    xmin = minimum(x)
    xmax = maximum(x)
    dx = xmax - xmin
    x_normalized = normalize_x(x, xmin, dx)
    # Compute chebyshev polynomials for the normalized x
    chebyshev_polys = compute_chebyshev_polynomials(x_normalized, polynomial_order)
    chebyshev_basis = cat(chebyshev_polys..., dims=1)
    # Compute polynomial output using polynomial weights
    poly_output = poly_weight(chebyshev_basis)
    # Combine base and polynomial outputs, normalize, and activate
    y = base_activation.(layer_norm(base_output .+ poly_output))
    return y
end

