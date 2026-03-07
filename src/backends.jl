
struct CPUBackend <: AbstractBackend

    platform::String
    device::String

    function CPUBackend()
        new("CPU", "CPU")
    end

end

struct cudaBackend <: AbstractGPUBackend

    platform::String
    device::String
    memtype::String
    memory::DataType

end

struct rocBackend <: AbstractGPUBackend

    platform::String
    device::String
    memtype::String
    memory::DataType

end

struct metalBackend <: AbstractGPUBackend

    platform::String
    device::String
    memtype::String
    memory::DataType

end
