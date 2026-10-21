module metalExt

    using Metal
    using Squiggles
    import Squiggles: set_GPUbackend, memcopy, used_backend
    using StyledStrings

    # Metal PrivateStorage = CUDA DeviceMemory, Metal SharedStorage = CUDA UnifiedMemory
    function set_GPUbackend(mem::String="device")

        if mem == "device"
            backend = Squiggles.metalBackend("GPU", "METAL", mem, Metal.PrivateStorage)

        elseif mem == "unified"
            backend = Squiggles.metalBackend("GPU", "METAL", mem, Metal.SharedStorage)

        else
            error(styled"Memory {bold:$mem} type not recognized")
        end
        @info styled"METAL with {bold:$(backend.memory)} will now be used."
        @info styled"{bold:NOTE: } Metal GPUs do not support Float64 (double) precision. Inputs with Float64 precision will be casted to Float32"
        @info styled"{bold:NOTE: } Multiple Metal GPUs are not supported"


        Squiggles.global_settings["backend"] = backend
        return nothing
    end


    function memcopy(A::AbstractArray{T, N}, dev_id::Int=0) where {T, N}
        mem = used_backend().memory

        # if T == Float64
        #     A = Float32.(A) #Downcast to Float32 since Metal does not support Float64
        # end

        NT = T == Float64 ? Float32 : T #Downcast to Float32 since Metal does not support Float64, else leave the type as is

        A_mtl = MtlArray{NT, N, mem}(A) # move to memory
        return A_mtl
    end

end
