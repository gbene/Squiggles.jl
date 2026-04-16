# Examples


Here are a couple of examples that can be run 


## 1. ``A\star B``

```julia

using Squiggles
using CUDA
using GLMakie

set_GPUbackend() # Set the backend to GPU

i = 10
j = 10

n_samples = 128

A = SignalMatrix(n_samples, 1, i, 1:20, 1, -0.5:0.1:0.5)
B = SignalMatrix(n_samples, 1, j, 1:20, 1, -0.5:0.1:0.5)

τ = 128

threads_per_block = 128

correlograms_gpu = correlogram(A, B, τ, threads_per_block) # Correlogram
correlograms_norm_gpu = norm_correlogram(A, B, τ, threads_per_block) # Normalized Correlogram
coeffs_gpu, lags = simplelags(correlograms_norm_gpu, τ) # Correlation coeffs and lags


correlograms = memcopy(correlograms_gpu) # Copy from GPU to CPU
correlograms_norm = memcopy(correlograms_norm_gpu)
coeffs = memcopy(coeffs_gpu)


corr_fig, corr_ax = plotCorrelogram(correlograms)
corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)
coeff_fig, coeff_ax = plotCC(coeffs)

display(corrn_fig)
```


## 2. ``A\star A``

```julia

using Squiggles
using CUDA
using GLMakie

set_GPUbackend() # Set the backend to GPU

i = 10

n_samples = 128

A = SignalMatrix(n_samples, 1, i, 1:20, 1, -0.5:0.1:0.5)

τ = 128

threads_per_block = 128

correlograms_gpu = correlogram(A, τ, threads_per_block) # Correlogram
correlograms_norm_gpu = norm_correlogram(A, τ, threads_per_block) # Normalized Correlogram
coeffs_gpu, lags = simplelags(correlograms_norm_gpu, τ) # Correlation coeffs and lags


correlograms = memcopy(correlograms_gpu) # Copy from GPU to CPU
correlograms_norm = memcopy(correlograms_norm_gpu)
coeffs = memcopy(coeffs_gpu)


corr_fig, corr_ax = plotCorrelogram(correlograms)
corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)
coeff_fig, coeff_ax = plotCC(coeffs)

display(corrn_fig)
```


## 3. Using a different GPU

To use a different GPU one can just change the imported julia package. For example to use Apple silicon the ``A\star A`` example is




```julia

using Squiggles
using Metal # <----- This changed
using GLMakie

set_GPUbackend() # Set the backend to GPU

i = 10

n_samples = 128

A = SignalMatrix(n_samples, 1, i, 1:20, 1, -0.5:0.1:0.5)

τ = 128

threads_per_block = 128

correlograms_gpu = correlogram(A, τ, threads_per_block) # Correlogram
correlograms_norm_gpu = norm_correlogram(A, τ, threads_per_block) # Normalized Correlogram
coeffs_gpu, lags = simplelags(correlograms_norm_gpu, τ) # Correlation coeffs and lags


correlograms = memcopy(correlograms_gpu) # Copy from GPU to CPU
correlograms_norm = memcopy(correlograms_norm_gpu)
coeffs = memcopy(coeffs_gpu)


corr_fig, corr_ax = plotCorrelogram(correlograms)
corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)
coeff_fig, coeff_ax = plotCC(coeffs)

display(corrn_fig)
```