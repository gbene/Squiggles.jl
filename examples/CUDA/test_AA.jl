using Squiggles
using CUDA
using GLMakie

set_GPUbackend()

n_A = 10

n_samples = 128

A = SignalMatrix(n_samples, 1, n_A, 1:20, 1, -0.5:0.1:0.5)

τ = 128

# display(calculate_memory(A,A,τ))
# display(calculate_memory(A,τ))

threads_per_block = 128

# correlograms_norm_full_gpu = norm_correlogram(A, A, τ, threads_per_block)
correlograms_norm_gpu = norm_correlogram(A, τ, threads_per_block)
coeffs_gpu, lags_gpu = simplelags(correlograms_norm_gpu, τ)

coeffs = memcopy(coeffs_gpu)
lags = memcopy(lags_gpu)


coeffs_symm = reconstruct_symmetric(coeffs)
lags_symm = reconstruct_antisymmetric(lags)


# correlograms = memcopy(correlograms_norm_full_gpu)
correlograms_norm = memcopy(correlograms_norm_gpu)


# corr_fig, corr_ax = plotCorrelogram(correlograms)
corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)



# coeffs = memcopy(coeffs_gpu)

coeff_fig, coeff_ax = plotCC(coeffs_symm)
