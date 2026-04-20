macro def(name, definition)
  return quote
      macro $(esc(name))()
          esc($(Expr(:quote, definition)))
      end
  end
end


@def kernel_template begin

    n_lags = τ*2 +1

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
        lag = (i-1)-τ
        # @print("$lag")
        # @inbounds lag_cache[i] = lag
        @inbounds correlograms[template_block_index, signal_block_index,i] = lagged_dot(template, signal, size_template, size_signal, lag)
    end
end

@def optimized_kernel_template begin

    n_lags = τ*2 +1

    size_template = size_signal = size(templates, 1)
    N = size(templates, 2)


    n, m = @index(Group, NTuple)
    linear_block_index = @index(Group, Linear)
    thread_index = @index(Local)
    stride = @groupsize()[1]



    template_block_index, signal_block_index = linear2tr_nodiag(linear_block_index, N)

    # if thread_index == 1
    #     @print("$linear_block_index, ($template_block_index, $signal_block_index)\n")
    # end
    end_templ_idx = size_template

    start_signal_idx = end_templ_idx+1
    end_signal_idx = end_templ_idx+size_template


    # We slice the shmem in the different things we need
    # view -> pointer to shmem memory address

    template = view(shmem, 1:end_templ_idx)
    signal = view(shmem, start_signal_idx:end_signal_idx)



    # Move the signal and template in the shared memory

    # @unroll
    @unroll for i = thread_index:stride:size_template
        @inbounds template[i] = templates[i, template_block_index]
    end

    @unroll for i = thread_index:stride:size_signal
        @inbounds signal[i] = templates[i, signal_block_index]
    end

    @synchronize()

    # Grid stride loop to calculate lags.
    # We use a grid stride so that we can use any number of lags independently from the number of lunched threads

    # @unroll
    @unroll for i = thread_index:stride:n_lags
        lag = (i-1)-τ
        # @print("$lag")
        # @inbounds lag_cache[i] = lag
        @inbounds correlograms[n, m, i] = lagged_dot(template, signal, size_template, size_template, lag)
    end
end
