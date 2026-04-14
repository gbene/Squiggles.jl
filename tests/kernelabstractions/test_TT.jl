using Squiggles
using CUDA
using GLMakie
using BenchmarkTools
include("kernels.jl")

set_GPUbackend()

function plotCC(cc_mat)

    fig = Figure(size=(900,900))
    ax = Axis(fig[1, 1], aspect=1, yreversed=true)
    heatmap!(ax, cc_mat)

    display(fig)

    return fig, ax

end

function plotCorrelogram(correlograms)

    perm_corrs = permutedims(correlograms,(3,1,2))
    n_signals, n_templates, n_lags = size(correlograms)
    corr_length = (n_lags-1)/2
    fig = Figure(size=(900,900))
    ax = Axis3(fig[1, 1], aspect=:equal, xlabel="Lags", ylabel="Signal index", zlabel="Template index")

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





n_templates = 1000
n_samples = 128

corr_length = 128 # How many samples are correlated
nlags = (corr_length*2)+1 # total number of lags being correlated (assuming power of 2 corr_length)

templates = SignalMatrix(n_samples, 1, n_templates, 1:20, 3, -2:0.25:2)
norm_templates = Squiggles.normalize_columns(templates)
correlograms = zeros(Float32, n_templates, n_templates, nlags)



nthreads = (corr_length,1) # Set number of threads per block. If = corr_length then one thread per "lag"
# nthreads = (16,1) # Set number of threads per block. If = corr_length then one thread per "lag"

blocks = (n_templates, n_templates) # Set number of blocks. This is n_templates x n_signals however in this case templates=signals
ndrange = nthreads .* blocks # Total number of threads that need to be launched i.e. nthreads*blocks

templates_gpu = memcopy(norm_templates)
signals_gpu = memcopy(norm_templates)
correlograms_gpu = memcopy(correlograms) # The correlogram matrix is n_templates x n_signals x nlags


kernel128 = correlogram_ak128(get_backend(templates_gpu))

kernel128(templates_gpu, signals_gpu, corr_length, correlograms_gpu, ndrange=ndrange, workgroupsize=nthreads)

cc_gpu = maximum(correlograms_gpu,dims=3)

cc = memcopy(cc_gpu)[:,:]

correlograms = memcopy(correlograms_gpu)

cc_fig, cc_ax = plotCC(cc)
corr_fig, corr_ax = plotCorrelogram(correlograms)

# Free up gpu memory

templates_gpu = nothing
signals_gpu = nothing
correlograms_gpu = nothing

GC.gc()
