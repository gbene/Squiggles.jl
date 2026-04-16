
"""
    correlogram(A, B, τ, threads_per_block)
    correlogram(A, τ, threads_per_block)


Calculate the correlogram values for two signal matrices A (n_samples x n_signals) and B (n_samples x m_signals).
Each signal in A is run across each signal in B. The lag interval is [-τ, τ]

The resulting output will be a correlogram matrix of size (n_signals x m_signals x τ*2+1).
Also one signal matrix can be used in the inputs. In such cases it is assumed
`correlogram(A, A, τ, threads_per_block)` thus the output should be symmetric. To avoid
extra computation then only the signal pairs of the lower triangle of the matrix will be outputted


### Arguments

    - `A::AbstractArray` -- n_samples x n_signals matrix
    - `B::AbstractArray` -- n_samples x m_signals matrix
    - `τ::Int` -- Lag length
    - `threads_per_block` -- Number of threads per block
"""
function correlogram(A::AbstractArray{T}, B::AbstractArray{T}, τ::Int, threads_per_block::Int) where T

    n_signals = size(A, 2)
    m_signals = size(B, 2)
    n_samples = size(A, 1)

    nlags = (τ * 2) + 1

    correlograms = zeros(T, n_signals, m_signals, nlags)


    nthreads = (threads_per_block, 1)
    blocks = size(correlograms)[1:2]

    ndrange = nthreads .* blocks # Total number of threads that need to be launched i.e. nthreads*blocks


    A_gpu = memcopy(A)
    B_gpu = memcopy(B)
    correlograms_gpu = memcopy(correlograms)

    kernel = get_kernel(n_samples)(get_backend(A_gpu))

    kernel(A_gpu, B_gpu, τ, correlograms_gpu, ndrange=ndrange, workgroupsize=nthreads)

    return correlograms_gpu

end


function correlogram(A::AbstractArray{T}, τ::Int, threads_per_block::Int) where T

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

    A_gpu = memcopy(A)
    correlograms_gpu = memcopy(correlograms)


    kernel = get_kernel(n_samples)(get_backend(A_gpu))

    kernel(A_gpu, τ, correlograms_gpu, ndrange=ndrange, workgroupsize=nthreads)

    return correlograms_gpu

end


"""
    norm_correlogram(A, B, τ, threads_per_block)
    norm_correlogram(A, τ, threads_per_block)


Calculate the normalized correlogram values for two signal matrices A (n_samples x n_signals) and B (n_samples x m_signals).
Each signal in A is run across each signal in B. The lag interval is [-τ, τ]

The resulting output will be a correlogram matrix of size (n_signals x m_signals x τ*2+1).
Also one signal matrix can be used in the inputs. In such cases it is assumed
`correlogram(A, A, τ, threads_per_block)` thus the output should be symmetric. To avoid
extra computation then only the signal pairs of the lower triangle of the matrix will be outputted



### Arguments

    - `A::AbstractArray` -- n_samples x n_signals matrix
    - `B::AbstractArray` -- n_samples x m_signals matrix
    - `τ::Int` -- Lag length
    - `threads_per_block` -- Number of threads per block
"""
function norm_correlogram(A::AbstractArray{T}, B::AbstractArray{T}, τ::Int, threads_per_block::Int) where T

    A_norm = normalize_columns(A)
    B_norm = normalize_columns(B)

    return correlogram(A_norm, B_norm, τ, threads_per_block)


end

function norm_correlogram(A::AbstractArray{T}, τ::Int, threads_per_block::Int) where T

    A_norm = normalize_columns(A)

    return correlogram(A_norm, τ, threads_per_block)


end


"""
    simplelags(correlograms, τ)


Find the correlation or anticorrelation coeff and correspective lag as index position. These values correspond to the absolute
maximum correlation value (preserving the original sign).

"""
function simplelags(correlograms::AbstractArray{T, 3}, τ::Int) where T
    idx = findmax(abs,correlograms, dims=3)[2] # get the index of the absolute maximums
    coeffs = correlograms[idx]
    lags = map(x -> x[3] .- τ, idx) # the lags are the index - the amount of lag used

    return coeffs, lags
end
