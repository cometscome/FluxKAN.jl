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
