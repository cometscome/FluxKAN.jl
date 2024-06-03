


mutable struct Radial_distribution_function
    grids::Vector{Float64}
    denominator::Float64
    num_grids::Int64
    grid_max::Float64
    grid_min::Float64
end

```
KAGnet: 
Gaussian version
```
mutable struct KAGnet{in_dim,out_dim,num_grids}
    base_weight
    poly_weight
    layer_norm
    base_activation
    in_dim::Int64
    out_dim::Int64
    num_grids::Int64
    rdf::Radial_distribution_function
end



function Radial_distribution_function(num_grids, grid_min, grid_max)
    grids = range(grid_min, grid_max, length=num_grids)
    denominator = (grid_max - grid_min) / (num_grids - 1)
    return Radial_distribution_function(grids, denominator, num_grids, grid_max, grid_min)
end
export Radial_distribution_function
Flux.@layer Radial_distribution_function trainable = ()


function (m::Radial_distribution_function)(x)
    y = rdf_foward(x, m.num_grids, m.grids, m.denominator)
end

function rdf_foward(x, num_grids, grids, denominator)
    y = []
    for n = 1:num_grids
        yn = exp.(-((x .- grids[n]) ./ denominator) .^ 2)
        push!(y, yn)
    end
    return y
end


function ChainRulesCore.rrule(::typeof(rdf_foward), x, num_grids, grids, denominator)
    y = []
    for n = 1:num_grids
        yn = exp.(-((x .- grids[n]) ./ denominator) .^ 2)
        push!(y, yn)
    end

    function pullback(ybar)
        sbar = NoTangent()
        dy = []
        for n = 1:num_grids
            dyn = (-2 .* (x .- grids[n]) ./ denominator^2) .* y[n]
            #dyn = 2 * (-((x .- grids[n]) ./ denominator)) * exp.(-((x .- grids[n]) ./ denominator) .^ 2)
            push!(dy, dyn)
        end
        dLdGdx = zero(x)
        for n = 1:length(ybar)
            dLdGdx .+= dy[n] .* ybar[n]
        end
        return sbar, dLdGdx, sbar, -dLdGdx, sbar
    end
    return y, pullback
end



function KAGnet(in_dim, out_dim; num_grids=8, base_activation=SiLU, grid_max=1, grid_min=-1)
    base_weight = Dense(in_dim, out_dim; bias=false)
    poly_weight = Dense(in_dim * num_grids, out_dim; bias=false)
    if out_dim == 1
        layer_norm = Dense(out_dim, out_dim; bias=false)
    else
        layer_norm = LayerNorm(out_dim)
    end
    rdf = Radial_distribution_function(num_grids, grid_min, grid_max)
    return KAGnet{in_dim,out_dim,num_grids}(base_weight,
        poly_weight, layer_norm, base_activation, in_dim, out_dim, num_grids, rdf)
end
function KAGnet(base_weight, poly_weight, layer_norm, base_activation, in_dim, out_dim, num_grids, rdf)
    return KAGnet{in_dim,out_dim,num_grids}(base_weight, poly_weight,
        layer_norm, base_activation,
        in_dim, out_dim, num_grids, rdf
    )
end

export KAGnet
Flux.@layer KAGnet

function (m::KAGnet{in_dim,out_dim,num_grids})(x) where {in_dim,out_dim,num_grids}
    y = KAGnet_forward(x, m.base_weight, m.poly_weight, m.layer_norm, m.base_activation, m.rdf)
end


function KAGnet_forward(x, base_weight, poly_weight, layer_norm, base_activation, rdf)
    # Apply base activation to input and then linear transform with base weights
    base_output = base_weight(base_activation.(x))
    # Normalize x to the range [-1, 1] for stable chebyshev polynomial computation
    xmin = minimum(x)
    xmax = maximum(x)
    dx = xmax - xmin
    x_normalized = normalize_x(x, xmin, dx)
    # Compute chebyshev polynomials for the normalized x
    chebyshev_polys = rdf(x_normalized)
    #compute_chebyshev_polynomials(x_normalized, polynomial_order)
    #chebyshev_polys = compute_chebyshev_polynomials(x_normalized, polynomial_order)
    chebyshev_basis = cat(chebyshev_polys..., dims=1)
    # Compute polynomial output using polynomial weights
    poly_output = poly_weight(chebyshev_basis)
    # Combine base and polynomial outputs, normalize, and activate
    y = base_activation.(layer_norm(base_output .+ poly_output))
    return y
end

