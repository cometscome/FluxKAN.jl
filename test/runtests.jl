using FluxKAN
using Test
using LegendrePolynomials
using Flux

function test()
    x = rand(3, 4)
    order = 4
    display(x)
    y = compute_legendre_polynomials(x, order)

    for n = 0:4
        yi = Pl.(x, n)
        @test yi == y[n+1]
        display(yi)
    end
end

function test2()
    x = rand(Float32, 3, 4)
    kan = KALnet(3, 2)
    #println(Flux.params(kan))
    #display(kan)
    y = kan(x)
    #display(y)
end

@testset "FluxKAN.jl" begin
    @testset "legendre_polynomials" begin
        # Write your tests here. 
        test()
    end
    @testset "KAN" begin
        # Write your tests here.
        test2()
    end
end


