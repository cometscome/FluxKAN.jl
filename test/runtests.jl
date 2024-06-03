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

    kan = KACnet(3, 2)
    #println(Flux.params(kan))
    #display(kan)
    y = kan(x)

    kan = KAGnet(3, 2)
    #println(Flux.params(kan))
    #display(kan)
    y = kan(x)
    #display(y)
end

function test3(method="L")
    n = 100
    x0 = range(-2, length=n, stop=2) #Julia 1.0.0以降はlinspaceではなくこの書き方になった。
    a0 = 3.0
    a1 = 2.0
    b0 = 1.0
    y0 = zeros(Float32, n)
    f(x0) = a0 .* x0 .+ a1 .* x0 .^ 2 .+ b0 .+ 3 * cos.(20 * x0)
    y0[:] = f.(x0)

    function make_φ(x0, n, k)
        φ = zeros(Float32, k, n)
        for i in 1:k
            φ[i, :] = x0 .^ (i - 1)
        end
        return φ
    end
    k = 4
    φ = make_φ(x0, n, k)
    #model = Dense(k, 1) #モデルの生成。W*x + b : W[1,k],b[1]
    #model = Chain(Dense(k, 10, relu), Dense(10, 1))
    if method == "L"
        model = Chain(KALnet(k, 10), KALnet(10, 1))
    elseif method == "C"
        model = Chain(KACnet(k, 10), KACnet(10, 1))
    elseif method == "G"
        model = Chain(KAGnet(k, 10), KAGnet(10, 1))
    elseif method == "GL"
        model = Chain(KAGLnet(k, 10), KAGLnet(10, 1))
    end
    display(model)
    #println("W = ", model[1].weight)
    #println("b = ", model[1].bias)

    loss(x, y) = Flux.mse(model(x), y) #loss関数。mseは平均二乗誤差
    opt = ADAM() #最適化に使う関数。ここではADAMを使用。

    function make_random_batch(x, y, batchsize)
        numofdata = length(y)
        A = rand(1:numofdata, batchsize)
        data = []
        for i = 1:batchsize
            push!(data, (x[:, A[i]], y[A[i]])) #ランダムバッチを作成。 [(x1,y1),(x2,y2),...]という形式
        end
        return data
    end

    function train_batch!(xtest, ytest, model, loss, opt, nt)
        for it = 1:nt
            data = make_random_batch(xtest, ytest, batchsize)
            Flux.train!(loss, Flux.params(model), data, opt)
            if it % 100 == 0
                lossvalue = 0.0
                for i = 1:length(ytest)
                    lossvalue += loss(xtest[:, i], ytest[i])
                end
                println("$(it)-th loss = ", lossvalue / length(y0))
            end
        end
    end

    batchsize = 20
    nt = 2000
    train_batch!(φ, y0, model, loss, opt, nt) #学習
    display(model)
    #println(model[1].weight) #W
    #println(model[1].bias) #b

end

function test4(method="L")
    function make_data(f)
        num = 47
        numt = 19
        numtrain = num * num
        numtest = numt * numt
        xtrain = range(-2, 2, length=num)
        ytrain = range(-2, 2, length=num)
        xtest = range(-2, 2, length=numt)
        ytest = range(-2, 2, length=numt)

        count = 0
        ztrain = Float32[]
        for i = 1:num
            for j = 1:num
                count += 1
                push!(ztrain, f(xtrain[i], ytrain[j]))
            end
        end

        count = 0
        ztest = Float32[]
        for i = 1:numt
            for j = 1:numt
                count += 1
                push!(ztest, f(xtest[i], ytest[j]))
            end
        end
        return xtrain, ytrain, ztrain, xtest, ytest, ztest
    end

    function make_inputoutput(x, y, z)
        count = 0
        numx = length(x)
        numy = length(y)
        input = zeros(Float64, 2, numx * numy)
        output = zeros(Float64, 2, numx * numy)
        count = 0
        for i = 1:numx
            for j = 1:numy
                count += 1
                input[1, count] = x[i]
                input[2, count] = y[j]
                output[1, count] = z[count]
            end
        end
        return input, output
    end


    function train_batch!(x_train, y_train, model, loss, opt_state, x_test, y_test, nepoch, batchsize)
        numtestdata = size(y_test)[2]
        train_loader = Flux.DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true)
        for it = 1:nepoch
            for (x, y) in train_loader
                grads = Flux.gradient(m -> loss(m(x), y), model)[1]
                Flux.update!(opt_state, model, grads)
            end

            if it % 10 == 0
                lossvalue = loss(model(x_test), y_test) / numtestdata
                println("$it-th testloss: $lossvalue")
            end
        end
    end


    function main(method="L")
        num = 30
        x = range(-2, 2, length=num)
        y = range(-2, 2, length=num)
        f(x, y) = x * y + cos(3 * x) + exp(y / 5) * x + tanh(y) * cos(3 * y)
        z = [f(i, j) for i in x, j in y]'
        #p = plot(x, y, z, st=:wireframe)
        #savefig("original.png")

        xtrain, ytrain, ztrain, xtest, ytest, ztest = make_data(f)

        input_train, output_train = make_inputoutput(xtrain, ytrain, ztrain)
        input_test, output_test = make_inputoutput(xtest, ytest, ztest)

        #train_loader = Flux.DataLoader((input_train,output_train), batchsize=5, shuffle=true);
        #model = Chain(Dense(2, 10, relu), Dense(10, 10, relu), Dense(10, 10, relu), Dense(10, 1))

        if method == "L"
            model = Chain(KALnet(2, 10), KALnet(10, 1))
        elseif method == "C"
            model = Chain(KACnet(2, 10), KACnet(10, 1))
        elseif method == "G"
            model = Chain(KAGnet(2, 10), KAGnet(10, 1))
        elseif method == "GL"
            model = Chain(KAGLnet(2, 10), KAGLnet(10, 1))
        end
        display(model)

        rule = Adam()
        opt_state = Flux.setup(rule, model)
        loss(y_hat, y) = sum((y_hat .- y) .^ 2)
        nepoch = 1000
        batchsize = 128
        train_batch!(input_train, output_train, model, loss, opt_state, input_test, output_test, nepoch, batchsize)

        znn = [model([i
            j])[1] for i in x, j in y]'
        #p = plot(x, y, [znn], st=:wireframe)
        #savefig("dense.png")
        display(model)

    end
    main(method)
end

@testset "FluxKAN.jl" begin
    @testset "legendre_polynomials" begin
        # Write your tests here. 
        test()
    end
    @testset "KAN" begin
        # Write your tests here.        
        test2()
        #=
        test3("L")
        test3("C")
        test3("G")
        test3("GL")
        println("test 4")
        println("KAL")
        test4("L")
        println("KAC")
        test4("C")
        println("KAG")
        test4("G")
        =#
        println("KAGL")
        test4("GL")
    end
end


