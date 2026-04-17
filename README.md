![](docs/src/assets/banner.svg)

Welcome to Squiggles.jl, a Julia package to accelerate 1D cross correlations of seismic signals using any GPU!

## Installation

Open the Julia REPL and type either 

```julia 
using Pkg; Pkg.add(url="git@github.com:gbene/Squiggles.jl.git")
```

or 

```julia 
] add git@github.com:gbene/Squiggles.jl.git 
```

To use GPU acceleration you need to install the appropriate JuliaGPU package

+ ```] add CUDA```      -- CUDA cards
+ ```] add Metal```     -- Apple silicon
+ More to come!



## Quick example

Here is a basic example using CUDA

```julia 
using Squiggles
using CUDA 

set_GPUbackend()

i = 10          # Number of signals (columns) for in the A matrix
j = 10          # Number of signals (columns) for in the B matrix

n_samples = 128 # Number of samples (rows) for A and B 

A = SignalMatrix(n_samples, 1, i, 1:20, 1, -0.5:0.1:0.5) # Random signals
B = SignalMatrix(n_samples, 1, j, 1:20, 1, -0.5:0.1:0.5)

τ = 128 # Amount of samples to correlate 

threads_per_block = 128 # Amount of threads per block 

correlograms_norm_gpu = norm_correlogram(A, B, τ, threads_per_block) # Normalized correlogram volume

norm_coeffs_gpu, lags_gpu = simplelags(correlograms_norm_gpu, τ) # Correlation coeffs and lags matrices
```

## Acknowledgments


### Libraries
We use many libraries of the Julia ecosystem and we thank all of them! These are the main ones that made this project possible

+ JuliaGPU: [https://juliagpu.org/](https://juliagpu.org/)
+ KernelAbstractions.jl: [https://juliagpu.github.io/KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/)

### People
+ Ylse Anna de Vries: [https://github.com/ylseanna](https://github.com/ylseanna)
+ Tom Winder: [https://github.com/TomWinder](https://github.com/TomWinder)
+ Elías Rafn Heimisson:  [https://github.com/eliasrh](https://github.com/eliasrh)