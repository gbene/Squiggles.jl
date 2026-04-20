"""
Benchmark in Julia
"""

using BenchmarkTools
using Squiggles
using CUDA
using JLD2
using Dates
using MAT

set_GPUbackend()

function foo(A, τ, threads_per_block)
    correlograms_norm_gpu = norm_correlogram(A, A, τ, threads_per_block)
    # coeffs_gpu, lags = simplelags(correlograms_norm_gpu, τ)
end

function foo!(correlograms, coeffs, lags, A, τ, threads_per_block)
    norm_correlogram!(correlograms, A, τ, threads_per_block)
    # simplelags!(coeffs, lags, correlograms, τ)
    return nothing
end
function foo!(correlograms, coeffs, lags, A, B, τ, threads_per_block)
    norm_correlogram!(correlograms, A, B, τ, threads_per_block)
    # simplelags!(coeffs, lags, correlograms, τ)
    return nothing
end

nsamples = 64
n_cols = 8000

mat = SignalMatrix(nsamples, 1, n_cols, 1:20, 1, -0.5:0.1:0.5) #data[4]
τ = 256



file = matopen("data/$(nsamples)_$(n_cols).mat", "w")
write(file, "mat", mat)
close(file)

# # A, B, correlograms, coeffs, lags = prepare_inputs(mat, mat, τ)
# A, correlograms, coeffs, lags = prepare_inputs(mat, τ)

# threads_per_block = 256

# # # correlograms_norm_full_gpu = norm_correlogram(A, A, τ, threads_per_block)
# # b1 = @btime CUDA.@sync foo(A, τ, threads_per_block)
# # CUDA.@time foo!(correlograms, coeffs, lags, A, B, τ, threads_per_block)
# foo!(correlograms, coeffs, lags, A, τ, threads_per_block)
# correlogramscpu = memcopy(correlograms)
# coeffscpu = memcopy(coeffs)
# lagscpu = memcopy(lags)

# CUDA.@time foo!(correlograms, coeffs, lags, A, τ, threads_per_block)



# # correlograms_norm = memcopy(correlograms_norm_gpu)


# # # corr_fig, corr_ax = plotCorrelogram(correlograms)
# # corrn_fig, corrn_ax = plotCorrelogram(correlograms_norm)

# A, correlograms, coeffs, lags = nothing, nothing, nothing, nothing

# GC.gc(true)
# CUDA.reclaim()

# # coeffs = memcopy(coeffs_gpu)

# coeff_fig, coeff_ax = plotCC(coeffs[:,:])
