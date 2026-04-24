module Squiggles

using GPUArrays
using StyledStrings
using LinearAlgebra
using KernelAbstractions
import KernelAbstractions.Extras.@unroll

include("base.jl")
include("utils.jl")
include("macros.jl")
include("device_functions.jl")
include("backends.jl")
include("signal_generators.jl")
include("kernels.jl")
# include("plotters.jl")
include("correlate.jl")


const supported_GPU_platforms  = ["CUDA"]
const supported_platforms = ["CPU", supported_GPU_platforms...]
const global_settings = Dict{String, Any}("backend"=>CPUBackend())

const kernels_dict = Dict{String, Any}(["16" => correlogram_ak16,
                                               "32" => correlogram_ak32,
                                               "64" => correlogram_ak64,
                                               "128" => correlogram_ak128,
                                               "256" => correlogram_ak256,
                                               "512" => correlogram_ak512,
                                               "1024" => correlogram_ak1024])

export used_backend, get_available_platforms, get_available_GPUplatforms, set_CPUbackend, set_GPUbackend, memcopy, get_kernel
export cc, ncc
export RandomEvent, AddPadding, AddNoise, ExtractSnippet, SignalMatrix
# export plotSignalMatrix, plotCorrelogram, plotCC
export normalize_columns, reduce_M_nodiag, calculate_memory, linear2tr_nodiag, reconstruct_symmetric, reconstruct_antisymmetric, prepare_inputs
export correlogram, norm_correlogram, simplelags, correlogram!, norm_correlogram!, simplelags!
end # module Squiggles
