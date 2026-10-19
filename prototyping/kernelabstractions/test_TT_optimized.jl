"""
Calculate the correlograms for autocorrelating a matrix of signals (i.e. do A⋆A). This is
a case specific application in which a couple assumptions can be made to be memory efficient.

1. templates (A) and signals (B) are the same, so only one input matrix can be used. This is
useful because less shared memory per block is used and thus more blocks/threads can be
launched.
2. Since the operation is A⋆A, the resulting correlogram matrix will be symmetric. This
means that we actually only need to run correlations for the lower (or upper) triangle. This
is useful because:
    + The correlogram matrix will occupy roughly half the memory
    + The correlation kernel processes only half of input, this means that technically it should take half the
    time!

"""


using Squiggles
using CUDA
using GLMakie
using BenchmarkTools
include("kernels.jl")

set_GPUbackend()

function plotCC(cc_mat)

    fig = Figure(size=(900,900))
    ax = Axis(fig[1, 1], yreversed=true)
    heatmap!(ax, cc_mat)

    display(fig)

    return fig, ax

end

function plotCorrelogram(correlograms)

    perm_corrs = permutedims(correlograms,(3,1,2))
    n_signals, n_templates, n_lags = size(correlograms)
    corr_length = (n_lags-1)/2
    fig = Figure(size=(900,900))
    ax = Axis3(fig[1, 1], xlabel="Lags", ylabel="Signal index", zlabel="Template index")

    sl_y = Slider(fig[2, 1], range = 1:n_signals, startvalue = 1)
    sl_z = Slider(fig[1, 2], range = 1:n_templates, startvalue = 1, horizontal=false)
    ax2 = Axis(fig[1,3], xlabel="Lags", ylabel="Normalized correlation", title="Normalized correlogram" )

    points = lift(sl_y.value, sl_z.value) do y, z
        [Point3f(-corr_length*1.1, y+0.5, z+0.5), Point3f(corr_length*1.1, y+0.5, z+0.5)]
    end

    text_pos = lift(sl_y.value, sl_z.value) do y, z
        Point3f(-corr_length*1.1, y+0.5, z+0.5)
    end

    text_label = lift(sl_y.value, sl_z.value) do y, z
        "$y,$z"
    end

    correlogram = lift(sl_y.value, sl_z.value) do y, z
        perm_corrs[:,y,z]
    end

    lines!(ax, points, color = :black, )
    textlabel!(ax, text_pos, text=text_label, fontsize = 15)
    lines!(ax2,correlogram)


    volume!(ax, (-corr_length,corr_length), (1,n_signals),(1,n_templates), perm_corrs, interpolate = false)

    display(fig)

    return fig, ax

end



function localpararellMax(cache::AbstractVector{T},
                     lag_cache::AbstractVector{T},
                     thread_index::Int32,
                     cache_length::Int32,
                     stride::Int32) where T

        # offset = div(cache_length, 2)
        s_power = exponent(cache_length)-1 # get the exponent number of the cachelength (power of 2)

        for p in s_power:-1:-1
            offset = 1<<p
            # We implement a grid stride loop to have reduction with any blockDim
            for i = thread_index:stride:cache_length
                if i <= offset
                    @inbounds val = cache[i]
                    @inbounds lag_val = lag_cache[i]
                    @inbounds offset_val = cache[i+offset]
                    @inbounds off_lag_val = lag_cache[i+offset]

                    if offset_val > val

                        @inbounds cache[i] = offset_val
                        @inbounds lag_cache[i] = off_lag_val
                    else
                        @inbounds cache[i] = val
                        @inbounds lag_cache[i] = lag_val
                    end
                end

            end
            @synchronize()
        end
end





n_templates = 10
n_samples = 128

corr_length = 128 # How many samples are correlated
nlags = (corr_length*2)+1 # total number of lags being correlated (assuming power of 2 corr_length)

templates = SignalMatrix(n_samples, 1, n_templates, 1:20, 3, -2:0.25:2)
norm_templates = Squiggles.normalize_columns(templates)
sz = (n_templates, n_templates, nlags)

if iseven(n_templates)

    sz = (n_templates-1, n_templates ÷ 2, nlags)

else

    sz = (n_templates, (n_templates - 1) ÷ 2, nlags)

end

correlograms = zeros(Float32, sz)
# correlograms2 = zeros(Float32, (n_templates, n_templates, nlags))



println("Template size: $(sizeof(templates)/1e9), Correlogram matrix size: $(sizeof(correlograms)/1e9)")

nthreads = (corr_length,1) # Set number of threads per block. If = corr_length then one thread per "lag"

blocks = size(correlograms)[1:2] # Set number of blocks. This is the size of the output matrix
# blocks2 = size(correlograms2)[1:2] # Set number of blocks. This is the size of the output matrix

ndrange = nthreads .* blocks # Total number of threads that need to be launched i.e. nthreads*blocks
# ndrange2 = nthreads .* blocks2 # Total number of threads that need to be launched i.e. nthreads*blocks

templates_gpu = memcopy(norm_templates)
correlograms_gpu = memcopy(correlograms) # The correlogram matrix is n_templates x n_signals x nlags
# correlograms_gpu2 = memcopy(correlograms2) # The correlogram matrix is n_templates x n_signals x nlags


kernel = Squiggles.correlogram_ak128(get_backend(templates_gpu))
kernel(templates_gpu, corr_length, correlograms_gpu, ndrange=ndrange, workgroupsize=nthreads)


# b = @benchmark (CUDA.@sync kernel(X, corr_length, Y, ndrange=ndrange, workgroupsize=nthreads)) setup=(X=$templates_gpu; Y=$correlograms_gpu)
# # kernel128(templates_gpu, templates_gpu, corr_length, correlograms_gpu2, ndrange=ndrange2, workgroupsize=nthreads)
# display(b)

# cc_gpu = argmax(abs.(correlograms_gpu),dims=3)

# b2 = @benchmark (CUDA.@sync maximum(X, dims=3)) setup=(X=$correlograms_gpu;)
# # cc_gpu2 = maximum(correlograms_gpu2,dims=3)
# display(b2)

# cc = memcopy(cc_gpu)[:,:]
# # cc2 = memcopy(cc_gpu2)[:,:]


# correlograms = memcopy(correlograms_gpu)

# cc_fig, cc_ax = plotCC(cc)
# # cc2_fig, cc2_ax = plotCC(cc2)

# corr_fig, corr_ax = plotCorrelogram(correlograms)

# display(cc_fig)
# # Free up gpu memory

# templates_gpu = nothing
# signals_gpu = nothing
# correlograms_gpu = nothing
# correlograms_gpu2 = nothing

# GC.gc()
