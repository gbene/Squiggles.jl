using Squiggles
using CUDA
using GLMakie
using BenchmarkTools

using KernelAbstractions
import KernelAbstractions.Extras.@unroll

function PlotTraces(traces_mat, spacing; disp=false)

    fig = Figure(size=(300,900))
    ax = Axis(fig[1,1], aspect=1/3)

    n_templates = size(traces_mat,2)

    transl = 0:spacing:spacing*n_templates-1

    series!(ax, templates'.+transl, solid_color=:black)

    if disp
        display(fig)
    end
    return fig, ax
end


function PlotTraces!(ax, traces_mat, spacing)

    n_templates = size(traces_mat,2)

    transl = 0:spacing:spacing*n_templates-1

    series!(ax, templates'.+transl, solid_color=:black)

end

function plotCC(cc_mat, signals)

    fig = Figure(size=(900,900))
    ax = Axis(fig[1, 1], aspect=1, yreversed=true)
    ax2 = Axis(fig[1, 2], aspect=1/3)


    heatmap!(ax, cc_mat)
    PlotTraces!(ax2, signals, 6)

    display(fig)

end


function locallagged_dot(
    template::AbstractVector{T},
    signal::AbstractVector{T},
    size_template::Int,
    size_signal::Int,
    lag::Int) where T

    dot_prod = T(0)

    for j = 1:size_template
        signal_index = lag+j

        if 0 < signal_index <= size_signal
            @inbounds s = signal[signal_index]
            @inbounds t = template[j]
            dot_prod += s*t

        end
    end

    return dot_prod
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



@kernel function cc_ak(templates::AbstractArray{T},
                       signals::AbstractArray{T},
                       cc_mat::AbstractArray{T},
                       lag_mat::AbstractArray{T},
                       nlags::Int32,
                       lag_len::Int32,
                       shmem_size::Int32) where {T}

        size_template = size(templates, 1)
        size_signal = size(signals, 1)


        # Define the shmem size. It is the length of lags
        template_block_index, signal_block_index = @index(Group, NTuple)

        thread_index = @index(Local)

        stride = @groupsize()[1]

        # if thread_index == 1
        #     @print("$(@index(Group, NTuple)), $stride\n")
        # end

        shmem = @localmem Float32 shmem_size


        # start_lag_idx = nlags+1
        # end_lag_idx = nlags+nlags

        # start_templ_idx = end_lag_idx+1
        # end_templ_idx = end_lag_idx+size_template

        # start_signal_idx = end_templ_idx+1
        # end_signal_idx = end_templ_idx+size_signal


        # # We slice the shmem in the different things we need
        # # view -> pointer to shmem memory address

        # cc_cache = view(shmem, 1:nlags)
        # lag_cache = view(shmem, start_lag_idx:end_lag_idx)
        # template = view(shmem, start_templ_idx:end_templ_idx)
        # signal = view(shmem, start_signal_idx:end_signal_idx)



        # # Move the signal and template in the shared memory

        # # @unroll
        # @unroll for i = thread_index:stride:size_template
        #     @inbounds template[i] = templates[i, template_block_index]
        # end

        # #@unroll
        # @unroll for i = thread_index:stride:size_signal
        #     @inbounds signal[i] = signals[i, signal_block_index]
        # end
        # @synchronize()

        # # Grid stride loop to calculate lags.
        # # We use a grid stride so that we can use any number of lags independently from the number of lunched threads

        # # @unroll
        # @unroll for i = thread_index:stride:nlags
        #     lag = (i-1)-lag_len
        #     @inbounds lag_cache[i] = lag
        #     @inbounds cc_cache[i] = locallagged_dot(template, signal, size_template, size_signal, lag)
        # end

        # @synchronize()
        # # sync_threads()

        # # Define the max value of the calculated lags
        # localpararellMax(cc_cache, lag_cache, thread_index, nlags, stride)

        # if thread_index == 1 # only the first thread is writing to the output

        #     if cc_cache[end] > cc_cache[1]
        #         cc_cache[1] = cc_cache[end]
        #         lag_cache[1] = lag_cache[end]
        #     end

        #     @inbounds cc_mat[template_block_index, signal_block_index] = cc_cache[1]
        #     @inbounds lag_mat[template_block_index, signal_block_index] = lag_cache[1]
        # end
end

n_templates = 2
sampling_freq = 100
duration = 1
freq_range = 1:20
phase_range = 0:0.1:duration
padding = 0.78
delay_range = -0.2:0.1:0.2


templates = zeros(Float32, 256, n_templates)

for i in 1:n_templates
      delay = rand(delay_range)
      event, domain = RandomEvent(freq_range, phase_range, [1], sampling_freq, duration, 10)
      trace, time_trace = AddPadding(event, sampling_freq, padding, delay)
    #   template, ttime = ExtractSnippet(trace, time_trace, sampling_freq, padding+delay+0.3,padding+delay+duration-0.3)

      templates[:,i] = real(trace)
end



templates_c = CuArray(templates)
signals_c = CuArray(templates)

cc_mat_c = CuArray(zeros(Float32, n_templates, n_templates))
lag_mat_c = CuArray(zeros(Float32, n_templates, n_templates))

nthreads = 256
lag_length = Int32(256)
nlags = Int32((lag_length*2)+1)
blocks = (n_templates, n_templates)

shmem = Int32(nlags*2+size(templates_c,1)+size(signals_c,1))

kernel = cc_ak(KernelAbstractions.get_backend(templates_c), 256, size(cc_mat_c))

kernel(templates_c, templates_c, cc_mat_c, lag_mat_c, nlags, lag_length, shmem)

# PlotTraces(templates, 6;disp=true)
# Squiggles.set_GPUbackend()

# # # data = rand(256, 100)

# # # cc_mat, lag = cc(data, data, 256, 256)
# cc_mat, lag = cc(templates, templates, 256, 256)
# b1 = @benchmark CUDA.@sync cc(templates, templates, 256, 256)
# display(b1)
# plotCC(cc_mat, templates)

# # heatmap(ncc_mat,axis=(aspect=DataAspect(), yreversed=true))
