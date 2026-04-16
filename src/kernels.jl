"""
    correlogram_ak16(templates, signals, τ, correlograms)
    correlogram_ak16(templates, τ, correlograms)


Calculate correlograms for templates and signals that have 16 samples

### Arguments

- `templates::AbstractArray` -- Array of templates. This must be an array of 16 x ntemplates (rows x cols)
- `signals::AbstractArray` -- Array of signals. This must be an array of 16 x nsignals (rows x cols)
- `τ::Int` -- Length of samples to be correlated [-τ; τ].
- `correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x τ*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

`τ` controls the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
τ*2+1. For example, using τ = 16 will slide the template from τ = -16 to τ = 16.

"""
@kernel function correlogram_ak16(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    # nsamples = 16
    shmem = @localmem Float32 16*2

    @kernel_template

end

@kernel function correlogram_ak16(templates::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 16*2

    @optimized_kernel_template

end

"""
    correlogram_ak32(templates, signals, τ, correlograms)
    correlogram_ak32(templates, τ, correlograms)


Calculate correlograms for templates and signals that have 32 samples

### Arguments

- `templates::AbstractArray` -- Array of templates. This must be an array of 32 x ntemplates (rows x cols)
- `signals::AbstractArray` -- Array of signals. This must be an array of 32 x nsignals (rows x cols)
- `τ::Int` -- Length of samples to be correlated [-τ; τ].
- `correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x τ*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

`τ` controls the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
τ*2+1. For example, using τ = 32 will slide the template from τ = -32 to τ = 32.

"""
@kernel function correlogram_ak32(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    # nsamples = 32
    shmem = @localmem Float32 32*2

    @kernel_template

end

@kernel function correlogram_ak32(templates::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 32*2

    @optimized_kernel_template

end

"""
    correlogram_ak64(templates, signals, τ, correlograms)
    correlogram_ak64(templates, τ, correlograms)


Calculate correlograms for templates and signals that have 64 samples

### Arguments

- `templates::AbstractArray` -- Array of templates. This must be an array of 64 x ntemplates (rows x cols)
- `signals::AbstractArray` -- Array of signals. This must be an array of 64 x nsignals (rows x cols)
- `τ::Int` -- Length of samples to be correlated [-τ; τ].
- `correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x τ*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

`τ` controls the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
τ*2+1. For example, using τ = 64 will slide the template from τ = -64 to τ = 64.

"""
@kernel function correlogram_ak64(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    # nsamples = 64
    shmem = @localmem Float32 64*2

    @kernel_template

end

@kernel function correlogram_ak64(templates::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 64*2

    @optimized_kernel_template

end




"""
    correlogram_ak128(templates, signals, τ, correlograms)
    correlogram_ak128(templates, τ, correlograms)


Calculate correlograms for templates and signals that have 128 samples

### Arguments

- `templates::AbstractArray` -- Array of templates. This must be an array of 128 x ntemplates (rows x cols)
- `signals::AbstractArray` -- Array of signals. This must be an array of 128 x nsignals (rows x cols)
- `τ::Int` -- Length of samples to be correlated [-τ; τ].
- `correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x τ*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

`τ` controls the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
τ*2+1. For example, using τ = 128 will slide the template from τ = -128 to τ = 128.

"""
@kernel function correlogram_ak128(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 128*2

    @kernel_template

end

@kernel function correlogram_ak128(templates::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 128*2

    @optimized_kernel_template

end

"""
    correlogram_ak256(templates, signals, τ, correlograms)
    correlogram_ak256(templates, τ, correlograms)


Calculate correlograms for templates and signals that have 256 samples

### Arguments

- `templates::AbstractArray` -- Array of templates. This must be an array of 256 x ntemplates (rows x cols)
- `signals::AbstractArray` -- Array of signals. This must be an array of 256 x nsignals (rows x cols)
- `τ::Int` -- Length of samples to be correlated [-τ; τ].
- `correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x τ*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

`τ` controls the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
τ*2+1. For example, using τ = 256 will slide the template from τ = -256 to τ = 256.

"""
@kernel function correlogram_ak256(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 256*2

    @kernel_template

end

@kernel function correlogram_ak256(templates::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 256*2

    @optimized_kernel_template

end

"""
    correlogram_ak512(templates, signals, τ, correlograms)
    correlogram_ak512(templates, τ, correlograms)


Calculate correlograms for templates and signals that have 512 samples

### Arguments

- `templates::AbstractArray` -- Array of templates. This must be an array of 512 x ntemplates (rows x cols)
- `signals::AbstractArray` -- Array of signals. This must be an array of 512 x nsignals (rows x cols)
- `τ::Int` -- Length of samples to be correlated [-τ; τ].
- `correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x τ*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

`τ` controls the total number of lags that will be performed for the correlation. The lags will be symmetrical in respect to τ = 0 and thus will be
τ*2+1. For example, using τ = 512 will slide the template from τ = -512 to τ = 512.

"""
@kernel function correlogram_ak512(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 512*2

    @kernel_template

end

@kernel function correlogram_ak512(templates::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 512*2

    @optimized_kernel_template

end

"""
    correlogram_ak1024(templates, signals, τ, correlograms)
    correlogram_ak1024(templates, τ, correlograms)


Calculate correlograms for templates and signals that have 1024 samples

### Arguments

- `templates::AbstractArray` -- Array of templates. This must be an array of 1024 x ntemplates (rows x cols)
- `signals::AbstractArray` -- Array of signals. This must be an array of 1024 x nsignals (rows x cols)
- `τ::Int` -- Length of samples to be correlated [-τ; τ].
- `correlograms::AbstractArray` -- Output array of correlograms. This must be of size ntemplates x nsignals x τ*2+1

### Output

This functions works inplace and will overwrite data in the `correlograms` matrix

### Notes

`τ` controls the total number of lags that will be performed for the correlation. The lags are symmetrical in respect to τ = 0 and thus will be
τ*2+1. For example, using τ = 1024 will slide the template from τ = -1024 to τ = 1024.

"""
@kernel function correlogram_ak1024(templates::AbstractArray{T},
                                   signals::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 1024*2

    @kernel_template

end

@kernel function correlogram_ak1024(templates::AbstractArray{T},
                                   τ::Int,
                                   correlograms::AbstractArray{T},
                                   ) where T


    shmem = @localmem Float32 1024*2

    @optimized_kernel_template

end
