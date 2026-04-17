using Squiggles
using CUDA
using GLMakie

set_GPUbackend()

n_A = 10

n_samples = 128

A = SignalMatrix(n_samples, 1, n_A, 1:20, 1, -0.5:0.1:0.5)

τ = 128


A_gpu, correlograms_norm_gpu, coeffs_gpu, lags_gpu = prepare_inputs(A, τ)


threads_per_block = 128

norm_correlogram!(correlograms_norm_gpu, A_gpu, τ, threads_per_block)
simplelags!(coeffs_gpu, lags_gpu, correlograms_norm_gpu, τ)

correlograms_norm = memcopy(correlograms_norm_gpu)

corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)
coeffs = memcopy(coeffs_gpu)

coeff_fig, coeff_ax = plotCC(coeffs)
