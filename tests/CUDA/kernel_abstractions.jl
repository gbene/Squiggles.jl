using KernelAbstractions, Test, Random
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

@kernel function naive_transpose_kernel!(a, b)
    i, j = @index(Global, NTuple)

    if @index(Local) == 1
    @print("$i, $j, $(@groupsize()), $(@index(Local))\n")
    end

    @inbounds a[i, j] = b[j, i]
end

# create wrapper function to check inputs
# and select which backend to launch on.
function naive_transpose!(a, b)
    if size(a)[1] != size(b)[2] || size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    backend = get_backend(a)
    @assert get_backend(b) == backend
    groupsize = KernelAbstractions.isgpu(backend) ? 256 : 1024
    kernel! = naive_transpose_kernel!(backend, 1, size(a))
    kernel!(a, b)
    return
end

# resolution of grid will be res*res
res = 2

# creating initial arrays
b = rand!(allocate(backend, Float32, res, res))
a = KernelAbstractions.zeros(backend, Float32, res, res)

naive_transpose!(a, b)
KernelAbstractions.synchronize(backend)
