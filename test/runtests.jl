using Squiggles
using Test
using StyledStrings


for (i,a) in enumerate(ARGS)
    if a ∉ Squiggles.supported_platforms
        @info styled"No available test for $a, not testing"
        deleteat!(ARGS, i)
    end
end


if "CUDA" in ARGS
    @info styled"Running {bold:CUDA} tests"
    include("CUDA_tests.jl")
end

if "Metal" in ARGS
    @info styled"Running {bold:Metal} tests"
    include("Metal_tests.jl")
end

if ("CUDA" ∉ ARGS) && ("Metal" ∉ ARGS)
    @info styled"No available test for this combination of inputs"

end
