
include("macros.jl")
using KernelAbstractions
import KernelAbstractions.Extras.@unroll


@def kernel_template begin

    n_lags = corr_length*2 +1

    size_template = size(templates, 1)
    size_signal = size(signals, 1)

    # Define the shmem size. It is the length of lags
    template_block_index, signal_block_index = @index(Group, NTuple)

    thread_index = @index(Local)

    stride = @groupsize()[1]

    end_templ_idx = size_template

    start_signal_idx = end_templ_idx+1
    end_signal_idx = end_templ_idx+size_signal


    # We slice the shmem in the different things we need
    # view -> pointer to shmem memory address

    template = view(shmem, 1:end_templ_idx)
    signal = view(shmem, start_signal_idx:end_signal_idx)



    # Move the signal and template in the shared memory

    # @unroll
    @unroll for i = thread_index:stride:size_template
        @inbounds template[i] = templates[i, template_block_index]
    end

    # @unroll
    @unroll for i = thread_index:stride:size_signal
        @inbounds signal[i] = signals[i, signal_block_index]
    end
    @synchronize()

    # Grid stride loop to calculate lags.
    # We use a grid stride so that we can use any number of lags independently from the number of lunched threads

    # @unroll
    @unroll for i = thread_index:stride:n_lags
        lag = (i-1)-corr_length
        # @print("$lag")
        # @inbounds lag_cache[i] = lag
        @inbounds correlograms[template_block_index, signal_block_index,i] = locallagged_dot(template, signal, size_template, size_signal, lag)
    end
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



# @kernel function correlogram_ak128(templates::AbstractArray{T},
#                                                 signals::AbstractArray{T},
#                                                 corr_length::Int,
#                                                 correlograms::AbstractArray{T},
#                                                 ) where T



#     nsamples = 128
#     n_lags = corr_length*2 +1
#     shmem = @localmem Float32 nsamples*2

#     size_template = size(templates, 1)
#     size_signal = size(signals, 1)

#     # Define the shmem size. It is the length of lags
#     template_block_index, signal_block_index = @index(Group, NTuple)

#     thread_index = @index(Local)

#     stride = @groupsize()[1]

#     end_templ_idx = size_template

#     start_signal_idx = end_templ_idx+1
#     end_signal_idx = end_templ_idx+size_signal


#     # We slice the shmem in the different things we need
#     # view -> pointer to shmem memory address

#     template = view(shmem, 1:end_templ_idx)
#     signal = view(shmem, start_signal_idx:end_signal_idx)



#     # Move the signal and template in the shared memory

#     # # @unroll
#     @unroll for i = thread_index:stride:size_template
#         @inbounds template[i] = templates[i, template_block_index]
#     end

#     # # #@unroll
#     @unroll for i = thread_index:stride:size_signal
#         @inbounds signal[i] = signals[i, signal_block_index]
#     end
#     @synchronize()

#     # # # Grid stride loop to calculate lags.
#     # # # We use a grid stride so that we can use any number of lags independently from the number of lunched threads

#     # # # @unroll
#     @unroll for i = thread_index:stride:n_lags
#         lag = (i-1)-corr_length
#         # @print("$lag")
#         # @inbounds lag_cache[i] = lag
#         @inbounds correlograms[template_block_index, signal_block_index,i] = locallagged_dot(template, signal, size_template, size_signal, lag)
#     end

#     # @synchronize()
# end

"""
    correlogram_ak16(templates, signals, corr_length, correlograms)

Calculate correlograms for templates and signals that have 16 samples

### Arguments

    -`templates::AbstractArray` -- Array of templates. This must be an array of 16 x ntemplates (rows x cols)
    -`signals::AbstractArray` -- Array of signals. This must be an array of 16 x nsignals (rows x cols)
    -`corr_length::Int` -- Length of samples to be correlated τ ∈ [-corr_length; corr_length].
    -`correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x corr_length*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

The `corr_length` the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
corr_length*2+1. For example, using corr_length = 16 will slide the template from τ = -16 to τ = 16.

"""
@kernel function correlogram_ak16(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   corr_length::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    # nsamples = 16
    shmem = @localmem Float32 16*2

    @kernel_template

end

"""
    correlogram_ak32(templates, signals, corr_length, correlograms)

Calculate correlograms for templates and signals that have 32 samples

### Arguments

    -`templates::AbstractArray` -- Array of templates. This must be an array of 32 x ntemplates (rows x cols)
    -`signals::AbstractArray` -- Array of signals. This must be an array of 32 x nsignals (rows x cols)
    -`corr_length::Int` -- Length of samples to be correlated τ ∈ [-corr_length; corr_length].
    -`correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x corr_length*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

The `corr_length` the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
corr_length*2+1. For example, using corr_length = 32 will slide the template from τ = -32 to τ = 32.

"""
@kernel function correlogram_ak32(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   corr_length::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    # nsamples = 32
    shmem = @localmem Float32 32*2

    @kernel_template

end

"""
    correlogram_ak64(templates, signals, corr_length, correlograms)

Calculate correlograms for templates and signals that have 64 samples

### Arguments

    -`templates::AbstractArray` -- Array of templates. This must be an array of 64 x ntemplates (rows x cols)
    -`signals::AbstractArray` -- Array of signals. This must be an array of 64 x nsignals (rows x cols)
    -`corr_length::Int` -- Length of samples to be correlated τ ∈ [-corr_length; corr_length].
    -`correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x corr_length*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

The `corr_length` the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
corr_length*2+1. For example, using corr_length = 64 will slide the template from τ = -64 to τ = 64.

"""
@kernel function correlogram_ak64(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   corr_length::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    # nsamples = 64
    shmem = @localmem Float32 64*2

    @kernel_template

end


"""
    correlogram_ak128(templates, signals, corr_length, correlograms)

Calculate correlograms for templates and signals that have 128 samples

### Arguments

    -`templates::AbstractArray` -- Array of templates. This must be an array of 128 x ntemplates (rows x cols)
    -`signals::AbstractArray` -- Array of signals. This must be an array of 128 x nsignals (rows x cols)
    -`corr_length::Int` -- Length of samples to be correlated τ ∈ [-corr_length; corr_length].
    -`correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x corr_length*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

The `corr_length` the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
corr_length*2+1. For example, using corr_length = 128 will slide the template from τ = -128 to τ = 128.

"""
@kernel function correlogram_ak128(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   corr_length::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    # nsamples = 128
    shmem = @localmem Float32 128*2

    @kernel_template

end
