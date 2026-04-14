module Squiggles

using GPUArrays
using StyledStrings
using Makie
using LinearAlgebra
# using KernelAbstractions

include("base.jl")
include("backends.jl")
include("signal_generators.jl")
include("device_functions.jl")
include("kernels.jl")
include("plotters.jl")
include("utils.jl")

const supported_GPU_platforms  = ["CUDA"]
const supported_platforms = ["CPU", supported_GPU_platforms...]
const global_settings = Dict{String, Any}("backend"=>CPUBackend())

export used_backend, get_available_platforms, get_available_GPUplatforms, set_CPUbackend, set_GPUbackend, memcopy
export cc, ncc
export RandomEvent, AddPadding, AddNoise, ExtractSnippet, SignalMatrix
export plotSignalMatrix
export normalize_columns
end # module Squiggles
