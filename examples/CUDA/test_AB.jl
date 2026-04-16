using Squiggles
using CUDA
using GLMakie

set_GPUbackend()

n_A = 10
m_B = 10

n_samples = 128

A = SignalMatrix(n_samples, 1, n_A, 1:20, 1, -0.5:0.1:0.5)
B = SignalMatrix(n_samples, 1, m_B, 1:20, 1, -0.5:0.1:0.5)

τ = 128

threads_per_block = 128

correlograms_gpu = correlogram(A, B, τ, threads_per_block)
correlograms_norm_gpu = norm_correlogram(A, B, τ, threads_per_block)


correlograms = memcopy(correlograms_gpu)
correlograms_norm = memcopy(correlograms_norm_gpu)


corr_fig, corr_ax = plotCorrelogram(correlograms)
corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)
