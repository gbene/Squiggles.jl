"""
Group all functions that can also be used in kernels (i.e. __device__ functions)
"""

"""
    lagged_dot(template::AbstractVector{T}, signal::AbstractVector{T}, size_template::Integer, size_signal::Integer, lag::Integer)

Calculate the dot product between the template and a lagged signal.
"""
function lagged_dot(
    template::AbstractVector{T},
    signal::AbstractVector{T},
    size_template::Integer,
    size_signal::Integer,
    lag::Integer) where T

    dot_prod = T(0)

    # This loop can't be unrolled I think, because of the if statement
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





# """
#     pararellMax(cache, lag_cache, thread_index, cache_length, stride)

# Calculate the maximum value using grid striding

# """
# function pararellMax end
