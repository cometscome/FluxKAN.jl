# FluxKAN: Julia version of the TorchKAN

[![Build Status](https://github.com/cometscome/FluxKAN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/FluxKAN.jl/actions/workflows/CI.yml?query=branch%3Amain)



This is a Julia version of the [TorchKAN](https://github.com/1ssb/torchkan). 
In the TorchKAN, 

> TorchKAN introduces a simplified KAN model and its variations, including KANvolver and KAL-Net, designed for high-performance image classification and leveraging polynomial transformations for enhanced feature detection.

In the original TorchKAN, the package uses the PyTorch. 
In the FluxKAN, this package uses the Flux.jl. 

I rewrote the TorchKAN with the Julia language. Now this package has 
- KAL-Net: Utilizing Legendre Polynomials in Kolmogorov Arnold Legendre Networks
only. 

# install
```
add https://github.com/cometscome/FluxKAN.jl
```

# MNIST

```julia
using FluxKAN
FluxKAN.MNIST_KAN()
```
or 
```julia
using FluxKAN
FluxKAN.MNIST_KAN(; batch_size=256, epochs=20, nhidden=64, polynomial_order=3)
```

## Author
Yuki Nagai, Ph. D.

Associate Professor in the Information Technology Center, The University of Tokyo.

## Contact
For support, please contact: nagai.yuki@mail.u-tokyo.ac.jp


## Cite this Project
If this project is used in your research or referenced for baseline results, please use the following BibTeX entries.

```bibtex
@misc{torchkan,
  author = {Subhransu S. Bhattacharjee},
  title = {TorchKAN: Simplified KAN Model with Variations},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/1ssb/torchkan/}}
}

@misc{fluxkan,
  author = {Yuki Nagai},
  title = {FluxKAN: Julia version of the TorchKAN},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cometscome/FluxKAN.jl}}
}
```
