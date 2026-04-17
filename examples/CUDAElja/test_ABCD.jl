using Squiggles
using CUDA

set_GPUbackend()

n_A = 10
m_B = 10

n_samples = 128

A = SignalMatrix(n_samples, 1, n_A, 1:20, 1, -0.5:0.1:0.5)
B = SignalMatrix(n_samples, 1, m_B, 1:20, 1, -0.5:0.1:0.5)
C = SignalMatrix(n_samples, 1, n_A, 1:20, 1, -0.5:0.1:0.5)
D = SignalMatrix(n_samples, 1, m_B, 1:20, 1, -0.5:0.1:0.5)

τ = 128

threads_per_block = 128

correlograms_norm_gpuAB = norm_correlogram(A, B, τ, threads_per_block)
correlograms_norm_gpuCD = norm_correlogram(C, D, τ, threads_per_block, device=1)


correlograms_normAB = memcopy(correlograms_norm_gpuAB)
correlograms_normCD = memcopy(correlograms_norm_gpuCD)
coeffs_gpuAB, lagsAB = simplelags(correlograms_norm_gpuAB, τ)
coeffs_gpuCD, lagsCD = simplelags(correlograms_norm_gpuCD, τ)
