using LinearAlgebra
using Random

"""
    linear2tr(k::Int, n::Int)

Get the i, j index of the lower triangle (including the diagonal) of a matrix of size n x n from a linear index k. Matrix is col-wise ordered

### Notes
https://atrebas.github.io/post/2021-01-17-index_to_lower_triangular_subscripts/
"""
function linear2tr(k::Int, n::Int)

    kp = n * (n + 1) ÷ 2 - k
    p = floor(Int, ( sqrt(1 + 8 * kp) - 1 ) / 2 )

    i = n - (kp - p * (p + 1) ÷ 2)
    j = n - p

    return CartesianIndex(i, j)

end


"""
    linear2tr_nodiag(k::Int, n::Int)

Get the i, j index of the lower triangle (excluding the diagonal) of a matrix of size n x n from a linear index k. Matrix is col-wise ordered

### Arguments

    -`k::Int` -- The linear index
    -`n::Int` -- The size of the matrix (n x n)

### Notes
https://atrebas.github.io/post/2021-01-17-index_to_lower_triangular_subscripts/
"""
function linear2tr_nodiag(k::Int, n::Int)

    kp = n * (n - 1) ÷ 2 - k
    p = floor(Int, ( sqrt(1 + 8 * kp) - 1 ) / 2 )

    i = n - (kp - p * (p + 1) ÷ 2)
    j = n - 1 - p

    return CartesianIndex(i, j)

end

"""
    diagonal_indexes(M)

Get the diagonal linear indexes of a NxN matrix
"""
function diagonal_indexes(M::Matrix{T}) where {T}

    N = size(M)[1]

    indexes = zeros(T, N)

    for k in 0:N-1
        indexes[k+1] = 1+k*(N+1)
    end

    return indexes
end

"""
    reduce_M

Given a symmetric matrix M of size N x N (with N > 2), get a reduced matrix m of the lower triangle, excluding the diagonal.
m will be of size N x (N+1)/2 (for N odd) or N/2 X (N+1) (for N even).


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

"""
    reconstruct_symm

Reconstruct a symmetrical matrix from a matrix representation of the lower triangle



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
N = 8 # size of the square matrix

n = N*(N+1) ÷ 2 # number of elements in the lower/upper triangle (including diagonal)
n_nodiag = N*(N-1) ÷ 2 # number of elements in the lower/upper triangle (excluding diagonal)


M = Matrix(reshape(1:N*N, (N,N)))

m = reduce_M_nodiag(M) # Reduce the symmetric matrix to the lower triangle (still as a matrix)
