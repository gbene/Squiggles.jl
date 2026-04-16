module cudaExt

    using CUDA
    using Squiggles
    import Squiggles: set_GPUbackend, memcopy, used_backend
    using StyledStrings

    function set_GPUbackend(mem::String="device")

        if mem == "device"
            backend = Squiggles.cudaBackend("GPU", "CUDA", mem, CUDA.DeviceMemory)

        elseif mem == "unified"
            backend = Squiggles.cudaBackend("GPU", "CUDA", mem, CUDA.UnifiedMemory)

        else
            error(styled"Memory {bold:$mem} type not recognized")
        end
        @info styled"CUDA with {bold:$(backend.memory)} will now be used"



        Squiggles.global_settings["backend"] = backend
        return nothing
    end

    function memcopy(A::AbstractArray{T, N}, dev_id::Int=0) where {T, N}
        mem = used_backend().memory

        prev_dev = device() # save current device

        if prev_dev.handle != dev_id

            device!(dev_id) # change to selected device

            A_cu = CuArray{T, N, mem}(A) # move to memory

            device!(prev_dev) # change back to starting device
            return A_cu
        else
            A_cu = CuArray{T, N, mem}(A) # move to memory
            return A_cu
        end
    end


    # function pararellMax(cache::AbstractVector{T},
    #                      lag_cache::AbstractVector{T},
    #                      thread_index::Int32,
    #                      cache_length::Int32,
    #                      stride::Int32) where T

    #     # offset = div(cache_length, 2)
    #     s_power = exponent(cache_length)-1 # get the exponent number of the cachelength (power of 2)

    #     for p in s_power:-1:-1
    #         offset = 1<<p
    #         # We implement a grid stride loop to have reduction with any blockDim
    #         for i = thread_index:stride:cache_length
    #             if i <= offset
    #                 @inbounds val = cache[i]
    #                 @inbounds lag_val = lag_cache[i]
    #                 @inbounds offset_val = cache[i+offset]
    #                 @inbounds off_lag_val = lag_cache[i+offset]

    #                 if offset_val > val

    #                     @inbounds cache[i] = offset_val
    #                     @inbounds lag_cache[i] = off_lag_val
    #                 else
    #                     @inbounds cache[i] = val
    #                     @inbounds lag_cache[i] = lag_val
    #                 end
    #             end

    #         end
    #         sync_threads() # We wait for all threads to finish

    #         # offset >>= 1
    #     end

    #     return

    # end

    # function cc_kernel(
    #     templates::CuDeviceArray{T},
    #     signals::CuDeviceArray{T},
    #     cc_mat::CuDeviceArray{T},
    #     lag_mat::CuDeviceArray{T},
    #     nlags::Int32,
    #     lag_len::Int32) where {T}

    #     size_template = size(templates, 1)
    #     size_signal = size(signals, 1)


    #     # Define the shmem size. It is the length of lags

    #     template_block_index = blockIdx().x
    #     signal_block_index = blockIdx().y

    #     thread_index = threadIdx().x
    #     stride = blockDim().x

    #     # @cuprintln("$(typeof(thread_index)), $(typeof(nlags)), $(typeof(stride))")

    #     # #The threads responsible for the lags are on the y

    #     # thread_index = threadIdx().y
    #     # stride = blockDim().y

    #     # The threads responsible for the dot product are on the x
    #     # dot_prod_stride = blockDim().x

    #     # The shared memory is the sum of:
    #     # 1. Number of lags x2 (one for the cc values one for the lag values)
    #     # (2. Number of threads doing the dot product)
    #     # 3. Size of the template + size of the signal

    #     # shmem = @cuDynamicSharedMem(T, nlags+dot_prod_stride+n_elements*2)
    #     shmem = @cuDynamicSharedMem(T, nlags*2+size_template+size_signal)


    #     # start_dot_idx =  nlags+1
    #     # end_dot_idx = nlags+dot_prod_stride

    #     # start_templ_idx = end_dot_idx+1
    #     # end_templ_idx = end_dot_idx+n_elements

    #     # start_signal_idx = end_templ_idx+1
    #     # end_signal_idx = end_templ_idx+n_elements

    #     start_lag_idx = nlags+1
    #     end_lag_idx = nlags+nlags

    #     start_templ_idx = end_lag_idx+1
    #     end_templ_idx = end_lag_idx+size_template

    #     start_signal_idx = end_templ_idx+1
    #     end_signal_idx = end_templ_idx+size_signal


    #     # We slice the shmem in the different things we need
    #     # view -> pointer to shmem memory address

    #     cc_cache = view(shmem, 1:nlags)
    #     lag_cache = view(shmem, start_lag_idx:end_lag_idx)
    #     template = view(shmem, start_templ_idx:end_templ_idx)
    #     signal = view(shmem, start_signal_idx:end_signal_idx)



    #     # Move the signal and template in the shared memory

    #     for i = thread_index:stride:size_template
    #         @inbounds template[i] = templates[i, template_block_index]
    #     end

    #     for i = thread_index:stride:size_signal
    #         @inbounds signal[i] = signals[i, signal_block_index]
    #     end
    #     sync_threads()

    #     # Grid stride loop to calculate lags.
    #     # We use a grid stride so that we can use any number of lags independently from the number of lunched threads

    #     for i = thread_index:stride:nlags
    #         lag = (i-1)-lag_len
    #         @inbounds lag_cache[i] = lag
    #         @inbounds cc_cache[i] =
    #             lagged_dot(template, signal, size_template, size_signal, lag)
    #     end


    #     sync_threads()

    #     # Define the max value of the calculated lags
    #     pararellMax(cc_cache, lag_cache, thread_index, nlags, stride)

    #     if thread_index == 1 # only the first thread is writing to the output

    #         if cc_cache[end] > cc_cache[1]
    #             cc_cache[1] = cc_cache[end]
    #             lag_cache[1] = lag_cache[end]
    #         end


    #         @inbounds cc_mat[template_block_index, signal_block_index] = cc_cache[1]
    #         @inbounds lag_mat[template_block_index, signal_block_index] = lag_cache[1]
    #     end
    #     return nothing

    # end

    # function ncc_kernel(
    #     templates::CuDeviceArray{T},
    #     signals::CuDeviceArray{T},
    #     ncc_mat::CuDeviceArray{T},
    #     lag_mat::CuDeviceArray{T},
    #     norm_templates::CuDeviceArray{T},
    #     norm_signals::CuDeviceArray{T},
    #     nlags::Int32,
    #     lag_len::Int32) where {T}

    #     size_template = size(templates, 1)
    #     size_signal = size(signals, 1)


    #     # Define the shmem size. It is the length of lags

    #     template_block_index = blockIdx().x
    #     signal_block_index = blockIdx().y

    #     thread_index = threadIdx().x
    #     stride = blockDim().x

    #     # #The threads responsible for the lags are on the y

    #     # thread_index = threadIdx().y
    #     # stride = blockDim().y

    #     # The threads responsible for the dot product are on the x
    #     # dot_prod_stride = blockDim().x

    #     # The shared memory is the sum of:
    #     # 1. Number of lags x2 (one for the cc values one for the lag values)
    #     # (2. Number of threads doing the dot product)
    #     # 3. Size of the template + size of the signal

    #     # shmem = @cuDynamicSharedMem(T, nlags+dot_prod_stride+n_elements*2)
    #     shmem = @cuDynamicSharedMem(T, nlags*2+size_template+size_signal)


    #     # start_dot_idx =  nlags+1
    #     # end_dot_idx = nlags+dot_prod_stride

    #     # start_templ_idx = end_dot_idx+1
    #     # end_templ_idx = end_dot_idx+n_elements

    #     # start_signal_idx = end_templ_idx+1
    #     # end_signal_idx = end_templ_idx+n_elements

    #     start_lag_idx = nlags+1
    #     end_lag_idx = nlags+nlags

    #     start_templ_idx = end_lag_idx+1
    #     end_templ_idx = end_lag_idx+size_template

    #     start_signal_idx = end_templ_idx+1
    #     end_signal_idx = end_templ_idx+size_signal


    #     # We slice the shmem in the different things we need
    #     # view -> pointer to shmem memory address

    #     cc_cache = view(shmem, 1:nlags)
    #     lag_cache = view(shmem, start_lag_idx:end_lag_idx)
    #     template = view(shmem, start_templ_idx:end_templ_idx)
    #     signal = view(shmem, start_signal_idx:end_signal_idx)



    #     # Move the signal and template in the shared memory

    #     for i = thread_index:stride:size_template
    #         @inbounds template[i] = templates[i, template_block_index]
    #     end

    #     for i = thread_index:stride:size_signal
    #         @inbounds signal[i] = signals[i, signal_block_index]
    #     end
    #     sync_threads()

    #     # Grid stride loop to calculate lags.
    #     # We use a grid stride so that we can use any number of lags independently from the number of lunched threads

    #     for i = thread_index:stride:nlags
    #         lag = (i-1)-lag_len
    #         @inbounds lag_cache[i] = lag
    #         @inbounds cc_cache[i] =
    #             lagged_dot(template, signal, size_template, size_signal, lag)
    #     end


    #     sync_threads()

    #     # Define the max value of the calculated lags
    #     pararellMax(cc_cache, lag_cache, thread_index, nlags, stride)

    #     if thread_index == 1 # only the first thread is writing to the output

    #         if cc_cache[end] > cc_cache[1]
    #             cc_cache[1] = cc_cache[end]
    #             lag_cache[1] = lag_cache[end]
    #         end

    #         norm = norm_templates[template_block_index]*norm_signals[signal_block_index]

    #         @inbounds ncc_mat[template_block_index, signal_block_index] = cc_cache[1]*norm
    #         @inbounds lag_mat[template_block_index, signal_block_index] = lag_cache[1]
    #     end
    #     return nothing

    # end


    # function cc(templates::AbstractArray{T}, signals::AbstractArray{T}, lag_length::Int, lag_threads::Int) where T

    #     lag_length = Int32(lag_length)
    #     lag_threads = Int32(lag_threads)

    #     templates_c = memcopy(templates)
    #     signals_c = memcopy(signals)

    #     length_templates = size(templates, 1) # all templates must have the same length
    #     length_signals = size(signals, 1) # all signals must have the same length


    #     n_templates = size(templates, 2)
    #     n_signals = size(signals, 2)


    #     cc_mat = CUDA.zeros(T, n_templates, n_signals)
    #     lag_mat = CUDA.zeros(T, n_templates, n_signals)

    #     nlags = Int32((lag_length*2)+1)

    #     threads = lag_threads

    #     blocks = (n_templates, n_signals)

    #     shmem = (nlags*2+length_templates+length_signals)*sizeof(T)

    #     @cuda threads=threads blocks=blocks shmem=shmem cc_kernel(
    #         templates_c,
    #         signals_c,
    #         cc_mat,
    #         lag_mat,
    #         nlags,
    #         lag_length,
    #     )

    #     mat_cpu = memcopy(cc_mat)
    #     lag_cpu = memcopy(lag_mat)

    #     return mat_cpu, lag_cpu

    # end

    # function ncc(templates::AbstractArray{T}, signals::AbstractArray{T}, lag_length::Int, lag_threads::Int) where T

    #     norm_values_templates = T.(1 ./ sqrt.(sum(templates .* templates, dims = 1)))
    #     norm_values_signals = T.(1 ./ sqrt.(sum(signals .* signals, dims = 1)))

    #     lag_length = Int32(lag_length)
    #     lag_threads = Int32(lag_threads)

    #     templates_c = memcopy(templates)
    #     signals_c = memcopy(signals)
    #     norm_values_templates_c = memcopy(norm_values_templates)
    #     norm_values_signals_c = memcopy(norm_values_signals)


    #     length_templates = size(templates, 1) # all templates must have the same length
    #     length_signals = size(signals, 1) # all signals must have the same length


    #     n_templates = size(templates, 2)
    #     n_signals = size(signals, 2)


    #     ncc_mat = memcopy(zeros(T, n_templates, n_signals))
    #     lag_mat = memcopy(zeros(T, n_templates, n_signals))

    #     nlags = Int32((lag_length*2)+1)

    #     threads = lag_threads

    #     blocks = (n_templates, n_signals)

    #     shmem = (nlags*2+length_templates+length_signals)*sizeof(T)

    #     @cuda threads=threads blocks=blocks shmem=shmem ncc_kernel(
    #         templates_c,
    #         signals_c,
    #         ncc_mat,
    #         lag_mat,
    #         norm_values_templates_c,
    #         norm_values_signals_c,
    #         nlags,
    #         lag_length,
    #     )

    #     mat_cpu = memcopy(ncc_mat)
    #     lag_cpu = memcopy(lag_mat)

    #     return mat_cpu, lag_cpu

    # end
end
