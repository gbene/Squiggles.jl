"""
    linear2tr_nodiag(k::Int, n::Int)

Get the i, j index of the lower triangle (excluding the diagonal) of a matrix of size n x n from a linear index k. Matrix is col-wise ordered

### Arguments

    -`k::Int` -- The linear index
    -`n::Int` -- The size of the matrix (n x n)

### Notes
https://atrebas.github.io/post/2021-01-17-index_to_lower_triangular_subscripts/
"""
function linear2tr_nodiag(k::Integer, n::Integer)

    kp = n * (n - 1) ÷ 2 - k
    p = floor(Int, ( sqrt(1 + 8 * kp) - 1 ) / 2 )

    i = n - (kp - p * (p + 1) ÷ 2)
    j = n - 1 - p

    return (i, j)

end


"""
    reduce_M(M)

Given a symmetric matrix **M** of size N x N (with N > 2), get a reduced matrix **m** of the lower triangle, excluding the diagonal.
**m** will be of size N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).


### Arguments
    -`M::Matrix` -- Input matrix of NxN

### Output

    -`m::Matrix` -- Reduced matrix of N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).

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
    reconstruct_nodiag(M)

Given a reduced matrix **m** of size N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd), get a symmetrical matrix **M** of size N x N.


### Arguments

    -`m::Matrix` -- Reduced matrix of N-1 x N/2 (for N even) or N X (N-1)/2 (for N odd).

### Output

    -`M::Matrix` -- Input matrix of NxN

"""

function reconstruct_nodiag(m::Matrix{T}) where T

    n_nodiag = length(m)

    N = (1+Int(sqrt(1+8*n_nodiag)))÷2
    M = zeros(T, N,N)

    M[diagonal_indexes(M)] .= 1

    for c in 1:n_nodiag
        i = linear2tr_nodiag(c, N)[1]
        j = linear2tr_nodiag(c, N)[2]

        M[i,j] = m[c]
        M[j,i] = m[c]
    end


    return M
end

"""
    normalize_columns(M)

Normalize the columns of matrix (x/norm(x) where x is the column)
"""
function normalize_columns(M::Matrix{T}) where T
    norm_rep(x) = x/norm(x)

    return mapslices(norm_rep, M, dims=1)

end



function calculate_memory(A::AbstractArray{T}, B::AbstractArray{T}, τ::Int) where T

    A_size = sizeof(A)
    B_size = sizeof(B)

    out_size = size(A,2) * size(B,2) * ((τ * 2) + 1) * sizeof(T)

    return Base.format_bytes(A_size+B_size+out_size)

end

function calculate_memory(A::AbstractArray{T}, τ::Int) where T

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
