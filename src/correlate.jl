
"""
    correlogram(A, B, τ, threads_per_block)
    correlogram(A, τ, threads_per_block)


Calculate the correlogram values for two signal matrices A (n_samples x n_signals) and B (n_samples x m_signals).
Each signal in A is run across each signal in B with lags ∈ [-τ, τ]


### Arguments

- `A::AbstractArray` -- n_samples x n_signals matrix
- `B::AbstractArray` -- n_samples x m_signals matrix
- `τ::Integer` -- Lag length
- `threads_per_block` -- Number of threads per block

### Outputs

- `correlograms::AbstractArray{T, 3}` -- The correlogram volume of size (n_signals x m_signals x τ*2+1).

### Notes

Also one signal matrix can be used as input. In such cases it is assumed A⋆A thus to avoid
extra computation, only correlations in the lower triangle will be carried out and the output will
be of size N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).

As of now only signals that have n samples as a power of 2 can be used. If this condition is not met an error is thrown


"""
function correlogram(A::AbstractArray{T}, B::AbstractArray{T}, τ::Integer, threads_per_block::Integer; device::Integer=0) where T
    assert_input(A)
    assert_input(B)

    n_signals = size(A, 2)
    m_signals = size(B, 2)
    n_samples = size(A, 1)


    nlags = (τ * 2) + 1

    correlograms = zeros(T, n_signals, m_signals, nlags)


    nthreads = (threads_per_block, 1)
    blocks = size(correlograms)[1:2]

    ndrange = nthreads .* blocks # Total number of threads that need to be launched i.e. nthreads*blocks


    A_gpu = memcopy(A, device)
    B_gpu = memcopy(B, device)
    correlograms_gpu = memcopy(correlograms, device)

    kernel = get_kernel(n_samples)(get_backend(A_gpu))

    kernel(A_gpu, B_gpu, τ, correlograms_gpu, ndrange=ndrange, workgroupsize=nthreads)

    return correlograms_gpu

end






function correlogram(A::AbstractArray{T}, τ::Integer, threads_per_block::Integer; device::Integer=0) where T
    assert_input(A)

    n_signals = size(A, 2)
    n_samples = size(A, 1)

    nlags = (τ * 2) + 1

    sz = (n_signals, n_signals, nlags)

    if iseven(n_signals)

        sz = (n_signals-1, n_signals ÷ 2, nlags)

    else

        sz = (n_signals, (n_signals - 1) ÷ 2, nlags)

    end

    correlograms = zeros(T, sz)


    nthreads = (threads_per_block, 1)
    blocks = size(correlograms)[1:2]

    ndrange = nthreads .* blocks # Total number of threads that need to be launched i.e. nthreads*blocks

    A_gpu = memcopy(A, device)
    correlograms_gpu = memcopy(correlograms, device)


    kernel = get_kernel(n_samples)(get_backend(A_gpu))

    kernel(A_gpu, τ, correlograms_gpu, ndrange=ndrange, workgroupsize=nthreads)

    return correlograms_gpu

end


"""
    correlogram!(correlograms, A, B, τ, threads_per_block)
    correlogram!(correlograms, A, τ, threads_per_block)

Inplace version of the [`correlogram`](@ref) function.

### Notes

This assumes that the inputs are already moved to the device, use [`prepare_inputs`](@ref)
"""
function correlogram!(correlograms::AbstractArray{T, 3}, A::AbstractArray{T}, B::AbstractArray{T}, τ::Integer, threads_per_block::Integer) where T
    assert_input(A)
    assert_input(B)


    n_samples = size(A, 1)

    nthreads = (threads_per_block, 1)
    blocks = size(correlograms)[1:2]

    ndrange = nthreads .* blocks # Total number of threads that need to be launched i.e. nthreads*blocks

    kernel = get_kernel(n_samples)(get_backend(A))

    kernel(A, B, τ, correlograms, ndrange=ndrange, workgroupsize=nthreads)

end

function correlogram!(correlograms::AbstractArray{T, 3}, A::AbstractArray{T}, τ::Integer, threads_per_block::Integer) where T
    assert_input(A)

    n_samples = size(A, 1)

    nthreads = (threads_per_block, 1)
    blocks = size(correlograms)[1:2]

    ndrange = nthreads .* blocks # Total number of threads that need to be launched i.e. nthreads*blocks

    kernel = get_kernel(n_samples)(get_backend(A))

    kernel(A, τ, correlograms, ndrange=ndrange, workgroupsize=nthreads)

end

"""
    norm_correlogram(A, B, τ, threads_per_block)
    norm_correlogram(A, τ, threads_per_block)


Calculate the normalized correlogram values for two signal matrices A (n_samples x n_signals) and B (n_samples x m_signals).
Each signal in A is run across each signal in B with lags ∈ [-τ, τ]


### Arguments

- `A::AbstractArray` -- n_samples x n_signals matrix
- `B::AbstractArray` -- n_samples x m_signals matrix
- `τ::Integer` -- Lag length
- `threads_per_block` -- Number of threads per block

### Outputs

- `correlograms::AbstractArray{T, 3}` -- The correlogram volume fo size (n_signals x m_signals x τ*2+1).

### Notes

Also one signal matrix can be used as input. In such cases it is assumed A⋆A thus to avoid
extra computation, only correlations in the lower triangle will be carried out and the output will
be of size N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).

As of now only signals that have n samples as a power of 2 can be used. If this condition is not met an error is thrown

"""
function norm_correlogram(A::AbstractArray{T}, B::AbstractArray{T}, τ::Integer, threads_per_block::Integer; device::Integer=0) where T
    assert_input(A)
    assert_input(B)

    A_norm = normalize_columns(A)
    B_norm = normalize_columns(B)

    return correlogram(A_norm, B_norm, τ, threads_per_block, device=device)


end

function norm_correlogram(A::AbstractArray{T}, τ::Integer, threads_per_block::Integer; device::Integer=0) where T
    assert_input(A)

    A_norm = normalize_columns(A)

    return correlogram(A_norm, τ, threads_per_block, device=device)


end


"""
    norm_correlogram!(correlograms, A, B, τ, threads_per_block)
    norm_correlogram!(correlograms, A, τ, threads_per_block)

Inplace version of the [`norm_correlogram`](@ref) function.

### Notes

This assumes that the inputs are already moved to the device, use [`prepare_inputs`](@ref)
"""
function norm_correlogram!(correlograms::AbstractArray{T, 3}, A::AbstractArray{T}, B::AbstractArray{T}, τ::Integer, threads_per_block::Integer) where T
    assert_input(A)
    assert_input(B)

    A_norm = normalize_columns(A)
    B_norm = normalize_columns(B)

    correlogram!(correlograms, A_norm, B_norm, τ, threads_per_block)

end

function norm_correlogram!(correlograms::AbstractArray{T, 3}, A::AbstractArray{T}, τ::Integer, threads_per_block::Integer) where T
    assert_input(A)

    A_norm = normalize_columns(A)

    correlogram!(correlograms, A_norm, τ, threads_per_block)


end


"""
    simplelags(correlograms, τ)
    simplelags(correlograms, τ, sampling_rate)


Find the correlation or anticorrelation coeffs and correspective lag. These values correspond to the absolute
maximum correlation value (preserving the original sign).
### Arguments

- `correlograms::AbstractArray{T, 3}` -- Correlogram volume
- `τ::Integer` -- lag length
- `sampling_rate::Integer` -- Sampling rate


### Outputs

- `coeffs::AbstractMatrix` -- Correlation coefficient matrix
- `lags::AbstractMatrix` -- Lag matrix

### Notes

If the sampling rate is provided then the lags will be in seconds


### Examples

- ```simplelags(correlograms, 128)``` -- Lags will be integers
- ```simplelags(correlograms, 128, 100)``` -- Lags will be in seconds

"""
function simplelags(correlograms::AbstractArray{T, 3}, τ::Integer) where T
    idx = findmax(abs, correlograms, dims=3)[2] # get the index of the absolute maximums
    coeffs = correlograms[idx]
    lags = map(x -> x[3] .- τ, idx) # the lags are the index - the amount of lag used

    return dropdims(coeffs, dims=3), dropdims(lags, dims=3)
end

function simplelags(correlograms::AbstractArray{T, 3}, τ::Integer, sampling_rate::Integer) where T

    coeffs, lags = simplelags(correlograms, τ)

    return coeffs, lags./sampling_rate
end


"""
    simplelags!(coeffs, lags, correlograms, τ)
    simplelags!(coeffs, lags, correlograms, τ, sampling_rate)

Inplace version of the [`simplelags`](@ref) function.

### Notes

This assumes that the inputs are already moved to the device, use [`prepare_inputs`](@ref)
"""
function simplelags!(coeffs::AbstractArray{T}, lags::AbstractArray{T}, correlograms::AbstractArray{T, 3}, τ::Integer) where T
    idx = findmax(abs, correlograms, dims=3)[2] # get the index of the absolute maximums
    coeffs .= correlograms[idx]
    map!(x -> x[3] .- τ, lags, idx) # the lags are the index - the amount of lag used

end

function simplelags!(coeffs::AbstractArray{T}, lags::AbstractArray{T}, correlograms::AbstractArray{T, 3}, τ::Integer, sampling_rate::Integer) where T

    simplelags!(coeffs, lags, correlograms, τ)

    lags ./= sampling_rate
end
