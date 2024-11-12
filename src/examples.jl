using Flux
using Flux: DataLoader
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using MLDatasets

function MNIST_KAN(; batch_size=256, epochs=20, nhidden=64, polynomial_order=3, method="Legendre")

    # Loading Dataset
    x_train, y_train = MLDatasets.MNIST.traindata(Float32)
    x_test, y_test = MLDatasets.MNIST.testdata(Float32)
    # Reshape Data in order to flatten each image into a linear array
    x_train = Flux.flatten(x_train) # 784×60000
    x_test = Flux.flatten(x_test) # 784×10000
    # One-hot-encode the labels
    y_train = onehotbatch(y_train, 0:9) # 10×60000
    y_test = onehotbatch(y_test, 0:9) # 10×10000

    img_size = (28, 28, 1)
    input_size = prod(img_size) # 784
    nclasses = 10 # 0~9
    # Define model
    #model = Chain(
    #    Dense(input_size, 32, relu),
    #    Dense(32, nclasses)
    #)
    nn = nhidden
    if method == "Legendre"
        model = Chain(
            KALnet(input_size, nn; polynomial_order),
            KALnet(nn, nclasses; polynomial_order)
        )
    elseif method == "Chebyshev"
        model = Chain(
            KACnet(input_size, nn; polynomial_order),
            KACnet(nn, nclasses; polynomial_order)
        )
    elseif method == "Gaussian"
        model = Chain(
            KAGnet(input_size, nn; num_grids=polynomial_order + 1),
            KAGnet(nn, nclasses; num_grids=polynomial_order + 1)
        )
    elseif method == "GaussianLearnable"
        model = Chain(
            KAGLnet(input_size, nn; num_grids=polynomial_order + 1),
            KAGLnet(nn, nclasses; num_grids=polynomial_order + 1)
        )
    else
        error("method = $medhod is not supported")
    end

    display(model)
    # parameter to be learned in the model
    parameters = Flux.params(model)

    # batch size and number of epochs
    #batch_size = 256
    #epochs = 30

    # Create minibatch loader for training and testing
    train_loader = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
    test_loader = DataLoader((x_test, y_test), batchsize=batch_size, shuffle=true)

    # Define optimizer
    opt = ADAM()

    # calculate loss for given data or collection of data
    function loss(x, y)
        ŷ = model(x)
        return logitcrossentropy(ŷ, y, agg=sum)
    end

    # calculate loss and accuracy of given collection of data
    function loss_accuracy(loader)
        acc = 0.0
        ls = 0.0
        num = 0
        for (x, y) in loader
            ŷ = model(x)
            ls += logitcrossentropy(ŷ, y, agg=sum)
            acc += sum(onecold(ŷ) .== onecold(y))
            num += size(x, 2)
        end
        return ls / num, acc / num
    end

    function callback(epoch)
        #display(model[2])
        println("Epoch=$epoch")
        train_loss, train_accuracy = loss_accuracy(train_loader)
        test_loss, test_accuracy = loss_accuracy(test_loader)
        println("    train_loss = $train_loss, train_accuracy = $train_accuracy")
        println("    test_loss = $test_loss, test_accuracy = $test_accuracy")

    end

    for epoch in 1:epochs
        Flux.train!(loss, parameters, train_loader, opt)
        callback(epoch)
    end

end

