module FluxKANMLDatasetsExt

using Flux
using Flux: DataLoader, onecold, onehotbatch
using Flux.Losses: logitcrossentropy
using FluxKAN
using MLDatasets

function FluxKAN.mnist_kan(
    ;
    batch_size=256,
    epochs=20,
    nhidden=64,
    polynomial_order=3,
    grid_size=5,
    spline_order=3,
    method="Legendre",
)
    train_data = MLDatasets.MNIST(split=:train)[:]
    test_data = MLDatasets.MNIST(split=:test)[:]
    x_train, y_train = train_data.features, train_data.targets
    x_test, y_test = test_data.features, test_data.targets
    x_train = Flux.flatten(x_train)
    x_test = Flux.flatten(x_test)
    y_train = onehotbatch(y_train, 0:9)
    y_test = onehotbatch(y_test, 0:9)

    input_size = 28 * 28
    nclasses = 10
    model = if method == "BSpline"
        Flux.Chain(
            FluxKAN.KANLinear(input_size, nhidden; grid_size, spline_order),
            FluxKAN.KANLinear(nhidden, nclasses; grid_size, spline_order),
        )
    elseif method == "Legendre"
        Flux.Chain(
            FluxKAN.KALnet(input_size, nhidden; polynomial_order),
            FluxKAN.KALnet(nhidden, nclasses; polynomial_order),
        )
    elseif method == "Chebyshev"
        Flux.Chain(
            FluxKAN.KACnet(input_size, nhidden; polynomial_order),
            FluxKAN.KACnet(nhidden, nclasses; polynomial_order),
        )
    elseif method == "Gaussian"
        Flux.Chain(
            FluxKAN.KAGnet(input_size, nhidden; num_grids=polynomial_order + 1),
            FluxKAN.KAGnet(nhidden, nclasses; num_grids=polynomial_order + 1),
        )
    elseif method == "GaussianLearnable"
        Flux.Chain(
            FluxKAN.KAGLnet(input_size, nhidden; num_grids=polynomial_order + 1),
            FluxKAN.KAGLnet(nhidden, nclasses; num_grids=polynomial_order + 1),
        )
    else
        throw(ArgumentError("method = $method is not supported"))
    end

    train_loader = DataLoader((x_train, y_train); batchsize=batch_size, shuffle=true)
    test_loader = DataLoader((x_test, y_test); batchsize=batch_size, shuffle=false)
    optimizer_state = Flux.setup(Flux.Adam(), model)

    function loss_accuracy(loader)
        accuracy_sum = 0
        loss_sum = 0.0
        sample_count = 0
        for (x, y) in loader
            prediction = model(x)
            loss_sum += logitcrossentropy(prediction, y; agg=sum)
            accuracy_sum += sum(onecold(prediction) .== onecold(y))
            sample_count += size(x, 2)
        end
        return loss_sum / sample_count, accuracy_sum / sample_count
    end

    for epoch in 1:epochs
        for (x, y) in train_loader
            gradients = Flux.gradient(model) do current_model
                logitcrossentropy(current_model(x), y; agg=sum)
            end[1]
            Flux.update!(optimizer_state, model, gradients)
        end

        train_loss, train_accuracy = loss_accuracy(train_loader)
        test_loss, test_accuracy = loss_accuracy(test_loader)
        println(
            "Epoch=$epoch train_loss=$train_loss train_accuracy=$train_accuracy " *
            "test_loss=$test_loss test_accuracy=$test_accuracy",
        )
    end

    train_loss, train_accuracy = loss_accuracy(train_loader)
    test_loss, test_accuracy = loss_accuracy(test_loader)
    return (;
        model,
        train_loss,
        train_accuracy,
        test_loss,
        test_accuracy,
    )
end

end
