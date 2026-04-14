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



function normalize_columns(M::Matrix{T}) where T
    norm_rep(x) = x/norm(x)

    return mapslices(norm_rep, M, dims=1)

end
