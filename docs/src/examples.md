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
coeffs_gpu, lags_gpu = simplelags(correlograms_norm_gpu, τ) # Correlation coeffs and lags


correlograms = memcopy(correlograms_gpu) # Copy from GPU to CPU
correlograms_norm = memcopy(correlograms_norm_gpu)
coeffs = memcopy(coeffs_gpu)
lags = memcopy(lags_gpu)

# When doing A⋆A, the output is in reduced form. To reconstruct it as a symmetric matrix do the follwowing

coeffs_symm = reconstruct_nodiag(coeffs)
lags_symm = reconstruct_nodiag(lags)

corr_fig, corr_ax = plotCorrelogram(correlograms)
corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)
coeff_fig, coeff_ax = plotCC(coeffs)
coeff_symm_fig, coeff_symm_ax = plotCC(coeffs_symm)

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
coeffs_gpu, lags_gpu = simplelags(correlograms_norm_gpu, τ) # Correlation coeffs and lags


correlograms = memcopy(correlograms_gpu) # Copy from GPU to CPU
correlograms_norm = memcopy(correlograms_norm_gpu)
coeffs = memcopy(coeffs_gpu)
lags = memcopy(lags_gpu)

# When doing A⋆A, the output is in reduced form. To reconstruct it as a symmetric matrix do the follwowing

coeffs_symm = reconstruct_nodiag(coeffs)
lags_symm = reconstruct_nodiag(lags)

corr_fig, corr_ax = plotCorrelogram(correlograms)
corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)
coeff_fig, coeff_ax = plotCC(coeffs)
coeff_symm_fig, coeff_symm_ax = plotCC(coeffs_symm)

display(corrn_fig)
```

## 4. Using multiple GPUs

Squiggles has the possibility of running the [correlogram](@ref) and [norm_correlogram](@ref) functions on multiple GPUs. This is useful for example when it is necessary to chunk a big input or process muliple inputs at once. This is still an experimental feature.


```julia

using Squiggles
using CUDA
using Dates



set_GPUbackend()

i = 5000
j = 5000
n_samples = 128
τ = 128
threads_per_block = 64

A = SignalMatrix(n_samples, 1, i, 1:20, 1, -0.5:0.1:0.5)
B = SignalMatrix(n_samples, 1, j, 1:20, 1, -0.5:0.1:0.5)
C = SignalMatrix(n_samples, 1, i, 1:20, 1, -0.5:0.1:0.5)
D = SignalMatrix(n_samples, 1, j, 1:20, 1, -0.5:0.1:0.5)

a_gpu, b_gpu, corrab_gpu, coeffab_gpu, lagsab_gpu = prepare_inputs(A, B, τ)
c_gpu, d_gpu, corrcd_gpu, coeffcd_gpu, lagscd_gpu = prepare_inputs(C, D, τ, device=1)




start = now()
@sync begin
    @async begin
        device!(0)
        norm_correlogram!(corrab_gpu, a_gpu, b_gpu, τ, threads_per_block)

    end
    @async begin
        device!(1)
        norm_correlogram!(corrcd_gpu, c_gpu, d_gpu, τ, threads_per_block)
    end
end
#


display("$(now()-start)")

corrab = memcopy(corrab_gpu)
corrcd = memcopy(corrcd_gpu)



a_gpu, b_gpu, corrab_gpu, coeffab_gpu, lagsab_gpu = nothing, nothing, nothing, nothing, nothing
c_gpu, d_gpu, corrcd_gpu, coeffcd_gpu, lagscd_gpu = nothing, nothing, nothing, nothing, nothing

device!(0)
GC.gc()
CUDA.reclaim()

device!(1)
GC.gc()
CUDA.reclaim()

```