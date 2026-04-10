module Squiggles

using GPUArrays
using StyledStrings
using Makie

include("base.jl")
include("backends.jl")
include("signal_generators.jl")
include("device_functions.jl")
include("kernels.jl")


const supported_GPU_platforms  = ["CUDA"]
const supported_platforms = ["CPU", supported_GPU_platforms...]
const global_settings = Dict{String, Any}("backend"=>CPUBackend())

export get_backend, get_available_platforms, get_available_GPUplatforms, set_CPUbackend, set_GPUbackend, cc, ncc, RandomEvent, AddPadding, AddNoise, ExtractSnippet

end # module Squiggles
