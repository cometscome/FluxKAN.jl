


mutable struct Radial_distribution_function
    grids#::Vector{Float64}
    denominator
    num_grids
    grid_max
    grid_min
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
    in_dim
    out_dim
    num_grids
    rdf::Radial_distribution_function
end



function Radial_distribution_function(num_grids, grid_min, grid_max)
    grids = range(grid_min, grid_max, length=num_grids)
    grids_W = Dense(1, num_grids; bias=false)
    #display(reshape(collect(grids), :, 1))
    #display(grids_W.weight)
    grids_W.weight .= reshape(collect(grids), :, 1)
    denominator = (grid_max - grid_min) / (num_grids - 1) |> f32
    return Radial_distribution_function(grids_W.weight, denominator, num_grids, grid_max, grid_min)
end
export Radial_distribution_function
Flux.@layer Radial_distribution_function trainable = ()


function (m::Radial_distribution_function)(x)
    y = rdf_foward(x, m.num_grids, m.grids, m.denominator)
end

function gauss_f(x, g, denominator)
    y = zero(x)
    @. y = exp(-((x - g) / denominator)^2)
    return y
end

function rdf_foward(x, num_grids, grids, denominator)
    y = []
    #y = map(g -> gauss_f(x,g,denominator),grids)
    #return y
    for n = 1:num_grids
        yn = zero(x)
        yn .= exp.(-((x .- grids[n]) ./ denominator) .^ 2)
        push!(y, yn)
    end
    return y
end


function ChainRulesCore.rrule(::typeof(rdf_foward), x, num_grids, grids, denominator)
    y = []
    for n = 1:num_grids
        yn = zero(x)
        yn .= exp.(-((x .- grids[n]) ./ denominator) .^ 2)
        push!(y, yn)
    end

    function pullback(ybar)
        sbar = NoTangent()

        dLdGdx = @thunk(begin
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
            dLdGdx
        end)
        dLdGdg = sbar
        #= note: not implemented now.
        @thunk(begin
            dy = []
            for n = 1:num_grids
                dyn = (2 .* (x .- grids[n]) ./ denominator^2) .* y[n]
                push!(dy, dyn)
            end
            dLdGdg = zero(grids)
            for n = 1:length(ybar)
                dLdGdg[n] = sum(dy[n] .* ybar, dims=1)
            end
            dLdGdg
        end)
        =#

        return sbar, dLdGdx, sbar, dLdGdg, sbar
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
    xt = base_activation.(x)
    base_output = base_weight(xt)
    #base_output = base_weight(base_activation.(x))
    # Normalize x to the range [-1, 1] for stable chebyshev polynomial computation
    xmin = minimum(x)
    xmax = maximum(x)
    dx = xmax - xmin
    if length(x) == 1
        x_normalized = x
    else
        x_normalized = normalize_x(x, xmin, dx)
    end
    #x_normalized = normalize_x(x, xmin, dx)
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

