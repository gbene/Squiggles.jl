abstract type AbstractBackend end
abstract type AbstractGPUBackend<:AbstractBackend end



"""
    memcopy(A::AbstractGPUArray)
    memcopy(A::AbstractArray, dev=0)

Copy an array to and from GPU/CPU depending on the output data type.

Overload this method with GPU specific Arrays using extensions

# Examples

```julia
using Squiggles
using CUDA

set_GPUbackend()

A = rand(256, 256)
memcopy(A) # Copy A to the default device of an available backend
memcopy(A, 1) # Copy A to device 1 of the available backend
```
"""
memcopy(A::AbstractGPUArray{T,N}) where {T, N} = Array{T, N}(A)


"""

    set_backend(backend::AbstractBackend)
Internal function used to set the backend used to perform the calculations.
Use the publicly available set_CPUbackend and set_GPUbackend to properly set the desired backend.
"""
function set_backend(backend::AbstractBackend)
    platform = backend.platform

    if platform in supported_platforms
        println(styled"{bold:$platform} will now be used")
        global_settings["backend"] = backend
    else
        error(styled"Platform {bold:$platform} is not supported")
    end
    return nothing
end

"""

    get_backend()

Return the current backend used to perform the calculations.

## Example

```julia
using Squiggles

get_backend()
```
```julia
using Squiggles
using CUDA

set_GPUbackend()

get_backend()
```
"""
function get_backend()
    return global_settings["backend"]
end


"""
    get_available_platforms()
Get the list of supported platforms that can be used for running the simulations

## Example

```julia
using Squiggles

get_available_platforms()
```
"""
get_available_platforms() = display(supported_platforms)


"""
    get_available_GPUplatforms()
Get the list of supported GPU platforms that can be used for running the simulations

## Example

```julia
using Squiggles

get_available_GPUplatforms()
```
"""
get_available_GPUplatforms() = display(supported_GPU_platforms)


"""
    set_CPUbackend()

Set the backend to CPU

# Example

```julia
using Squiggles
using CUDA

set_CPUbackend()
```
"""
set_CPUbackend() = set_backend(CPUBackend())



"""
    set_GPUbackend()

Set the backend to the available GPU.

## Notes

The set backend is decided by the installed packages, i.e. if CUDA.jl is
installed (and imported) then CUDA will be used. Only import one JuliaGPU package per script.
**If no JuliaGPU package is loaded, this method will return a LoadError.**
See the available GPU platforms by running get_available_GPUplatforms().
Some GPU backends support UnifiedMemory (e.g. CUDA, ROCm, Metal). The default will always be DeviceMemory but
users can choose UnifiedMemory by running ```set_GPUbackend("unified")```

## Example

```julia
using Squiggles
using CUDA

set_GPUbackend()
```
"""
function set_GPUbackend end

function cc end
function ncc end
