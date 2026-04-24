![icon](assets/banner.svg)


Welcome to Squiggles! 
Squiggles.jl is a Julia package to accelerate 1D cross correlations of seismic signals using GPUs! 
We use KernelAbstractions.jl to run on any GPU supported by the JuliaGPU ecosystem. There are however some caveats, please refer to the [Code structure and design](@ref) page of the docs.  


## Installation

### New to Julia?

If you are new to Julia here is a quick setup guide to get you started!

#### Install Julia

[Click here](https://julialang.org/downloads/) for the full guide on the Julia website

If on macOS or Linux type on a terminal

```bash
curl -fsSL https://install.julialang.org | sh
```

If on Windows, either download the [installer](https://install.julialang.org/Julia.appinstaller) or run the following

```bash
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
```

To check if the installation worked, type ```julia``` in a terminal. The following should appear

```bash

   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.12.3 (2025-12-15)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org release
|__/                   |

julia> 
```


#### Access the package manager

The REPL is an interactive programming environment that can also be used to install new packages. To access the REPL type ```julia``` in the terminal.
To access the package manager type ```]```. The input line should change from 

```julia
julia>
```
to

```julia
(@v1.12) pkg>
```

(depending on the version that you have the @v1.12 will change).

### Install Squiggles

Open the Julia REPL, enter the package manager using ```]``` and type

```julia 
add git@github.com:gbene/Squiggles.jl.git 
```

### Install GPU backends

To use GPU acceleration, you need to install the appropriate JuliaGPU package depending on the card that you have available. As of now only CUDA.jl and Metal.jl (Apple Silicon) are available. To install type 

```julia 
add CUDA
```

or

```julia 
add Metal
```



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






