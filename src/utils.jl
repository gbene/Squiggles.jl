"""
    diagonal_indexes(M)

Get the diagonal linear indexes of a NxN matrix
"""
function diagonal_indexes(M::Matrix{T}) where {T}

    N = size(M)[1]

    indexes = zeros(Int32, N)

    for k in 0:N-1
        indexes[k+1] = 1+k*(N+1)
    end

    return indexes
end


"""
    linear2tr_nodiag(k, n)

Get the i, j index of the lower triangle (excluding the diagonal) of a matrix of size n x n from a linear index k. Matrix is col-wise ordered

### Arguments

- `k::Integer` -- The linear index
- `n::Integer` -- The size of the matrix (n x n)

### Notes
https://atrebas.github.io/post/2021-01-17-index_to_lower_triangular_subscripts/
"""
function linear2tr_nodiag(k::Integer, n::Integer)

    kp = n * (n - 1) ÷ 2 - k
    p = floor(Integer, ( sqrt(1 + 8 * kp) - 1 ) / 2 )

    i = n - (kp - p * (p + 1) ÷ 2)
    j = n - 1 - p

    return (i, j)

end


"""
    reduce_M_nodiag(M)

Given a symmetric matrix **M** of size N x N (with N > 2), get a reduced matrix **m** of the lower triangle, excluding the diagonal.
**m** will be of size N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).


### Arguments

- `M::Matrix` -- Input matrix of NxN

### Output

- `m::Matrix` -- Reduced matrix of N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).

"""
function reduce_M_nodiag(M::Matrix{T}) where {T}
    N = size(M)[1]

    n_nodiag = N*(N-1) ÷ 2 # number of elements in the lower/upper triangle (excluding diagonal)

    m = zeros(T, n_nodiag)

    for i in 1:n_nodiag
        m[i] = M[linear2tr_nodiag(i, N)]
    end

    if iseven(N)
        m = reshape(m, N-1, N ÷ 2)
    else
        m = reshape(m, N, (N - 1)÷ 2)
    end


    return m

end

# function reduce_M_nodiag(M::Array{T, 3}) where {T}
#     N = size(M)[1]
#     L = size(M)[3]

#     n_nodiag = (N*(N-1) ÷ 2)*L # number of elements in the lower/upper triangle (excluding diagonal)

#     m = zeros(T, n_nodiag)

#     for i in 1:n_nodiag
#         m[i] = M[linear2tr_nodiag(i, N)]
#     end

#     if iseven(N)
#         m = reshape(m, N-1, N ÷ 2)
#     else
#         m = reshape(m, N, (N - 1)÷ 2)
#     end


#     return m

# end



"""
    reconstruct_symmetric(M; diagonal_values=1)

Given a reduced matrix **m** of size N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd), get a symmetrical matrix **M** of size N x N.


### Arguments

- `m::Matrix` -- Reduced matrix of N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).
- `diagonal_values` -- Number used to fill the diagonal. Default is 1

### Output

- `M::Matrix` -- Input matrix of NxN

### Notes

This function is mainly used after cross-correlation (A⋆A) to reconstruct a square matrix representation of the **cross correlation coefficients** in a lower triangular matrix form.
Because of this, the default value for filling the diagonal is 1.
"""
function reconstruct_symmetric(m::Matrix{T}; diagonal_values=1) where T

    n_nodiag = length(m)

    N = (1+Integer(sqrt(1+8*n_nodiag)))÷2
    M = zeros(T, N,N)

    M[diagonal_indexes(M)] .= diagonal_values

    for c in 1:n_nodiag
        i = linear2tr_nodiag(c, N)[1]
        j = linear2tr_nodiag(c, N)[2]

        M[i,j] = m[c]
        M[j,i] = m[c]
    end


    return M
end

"""
    reconstruct_antisymmetric(M; diagonal_values=0)

Given a reduced matrix **m** of size N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd), get a anti-symmetrical matrix **M** of size N x N.


### Arguments

- `m::Matrix` -- Reduced matrix of N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).
- `diagonal_values` -- Number used to fill the diagonal. Default is 0


### Output

- `M::Matrix` -- Input matrix of NxN

### Notes

This function is mainly used after cross-correlation (A⋆A) to reconstruct a square matrix representation of the **lags** in a lower triangle matrix form.
Because of this, the default value for filling the diagonal is 0.

"""
function reconstruct_antisymmetric(m::Matrix{T}; diagonal_values=0) where T

    n_nodiag = length(m)

    N = (1+Integer(sqrt(1+8*n_nodiag)))÷2
    M = zeros(T, N,N)

    M[diagonal_indexes(M)] .= diagonal_values

    for c in 1:n_nodiag
        i = linear2tr_nodiag(c, N)[1]
        j = linear2tr_nodiag(c, N)[2]

        M[i,j] = m[c]
        M[j,i] = m[c]*-1
    end


    return M
end


"""
    normalize_columns(M)

Normalize the columns of a matrix M (x/norm(x) where x is the column)
"""
function normalize_columns(M::AbstractMatrix{T}) where T
    norm_rep(x) = x/norm(x)

    return mapslices(norm_rep, M, dims=1)

end


"""
    calculate_memory(A, B, τ)
    calculate_memory(A, τ)


Calculate the memory required to perform A⋆B or A⋆A
"""
function calculate_memory(A::AbstractArray{T}, B::AbstractArray{T}, τ::Integer) where T

    A_size = sizeof(A)
    B_size = sizeof(B)

    out_size = size(A,2) * size(B,2) * ((τ * 2) + 1) * sizeof(T)

    return Base.format_bytes(A_size+B_size+out_size)

end

function calculate_memory(A::AbstractArray{T}, τ::Integer) where T

    A_size = sizeof(A)

    n_signals = size(A,2)
    nlags = ((τ * 2) + 1)

    sz = (n_signals, n_signals, nlags)

    if iseven(n_signals)

        sz = (n_signals-1, n_signals ÷ 2, nlags)

    else

        sz = (n_signals, (n_signals - 1) ÷ 2, nlags)

    end

    out_size = prod(sz) * sizeof(T)

    return Base.format_bytes(A_size+out_size)

end



function assert_input(A::AbstractArray)

    @assert ispow2(size(A,1)) "Cannot use signal lengths that are not power of 2!"

end

"""
    prepare_inputs(A, B, τ; device=0)
    prepare_inputs(A, τ; device=0)

Preallocate the inputs for [`correlogram!`](@ref), [`norm_correlogram!`](@ref) and [`simplelags!`](@ref) on device.

### Arguments

- `A::AbstractArray{T}` -- Signal matrix A
- `B::AbstractArray{T}` -- Signal matrix B
- `τ::Integer` -- Lag length
- `device::Integer` -- Device to copy the data

### Outputs

- `A_gpu::AbstractGPUArray{T}` -- Signal matrix A pointer on the device
- `B_gpu::AbstractGPUArray{T}` -- Signal matrix B pointer on the device
- `c_GPU::AbstractGPUArray{T, 3}` -- Empty correlogram volume pointer on the device
- `coeffs_GPU::AbstractGPUArray{T}` -- Empty correlation coefficient matrix pointer on the device
- `lags_GPU::AbstractGPUArray{T}` -- Empty lag matrix pointer on the device
"""
function prepare_inputs(A::AbstractArray{T}, B::AbstractArray{T}, τ::Integer; device::Integer=0) where T

    assert_input(A)
    assert_input(B)


    n_signals = size(A, 2)
    m_signals = size(B, 2)


    nlags = (τ * 2) + 1

    correlograms = zeros(T, n_signals, m_signals, nlags)

    coeffs = zeros(T, n_signals, m_signals)
    lags   = zeros(T, n_signals, m_signals)

    A_gpu = memcopy(A, device)
    B_gpu = memcopy(B, device)
    c_GPU = memcopy(correlograms, device)
    coeffs_GPU = memcopy(coeffs, device)
    lags_GPU = memcopy(lags, device)

    return A_gpu, B_gpu, c_GPU, coeffs_GPU, lags_GPU


end

function prepare_inputs(A::AbstractArray{T}, τ::Integer; device::Integer=0) where T
    assert_input(A)

    n_signals = size(A, 2)


    nlags = (τ * 2) + 1

    sz = (n_signals, n_signals, nlags)

    if iseven(n_signals)

        sz = (n_signals-1, n_signals ÷ 2, nlags)

    else

        sz = (n_signals, (n_signals - 1) ÷ 2, nlags)

    end

    correlograms = zeros(T, sz)

    coeffs = zeros(T, sz[1:2])
    lags   = zeros(T, sz[1:2])

    A_gpu = memcopy(A, device)
    c_GPU = memcopy(correlograms, device)
    coeffs_GPU = memcopy(coeffs, device)
    lags_GPU = memcopy(lags, device)

    return A_gpu, c_GPU, coeffs_GPU, lags_GPU


end
