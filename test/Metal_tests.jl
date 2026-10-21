using Squiggles
using JLD2
using Test
using Metal

set_GPUbackend()
check_data = load("$(@__DIR__)/files/1024.jld2")
@info styled"Using saved correlation data from {bold: $(check_data[\"date_created\"])}"

A = check_data["A"]
og_coeff = check_data["coeff"]
og_lags = check_data["lags"]

if typeof(A) == Float64 # Cast to Float32 if Float64 is used
    A = Float32.(A)
end
if typeof(og_coeff) == Float64 # Cast to Float32 if Float64 is used
    og_coeff = Float32.(og_coeff)
end


# Test that the results are the same as the one saved.
# We use directly norm_correlogram as at the end uses the correlogram function
@testset "Correlation tests" begin
    ncorr = norm_correlogram(A, A, 1024, 1024)
    ncorr_eff = norm_correlogram(A, 1024, 1024)
    coeff, lags = simplelags(ncorr, 1024)
    coeff_eff, lags_eff = simplelags(ncorr_eff, 1024)

    # Move to CPU
    coeff = memcopy(coeff)
    lags = memcopy(lags)
    coeff_eff = memcopy(coeff_eff)
    lags_eff = memcopy(lags_eff)

    @test isapprox(coeff, coeff') # Test that the coeff matrix is symmetric
    @test coeff == og_coeff # Test that the coeff matrix is the same as the one saved
    @test lags == og_lags # Test that the lag matrix is the same as the one saved
    @test isapprox(reconstruct_symmetric(coeff_eff, fillwithnans=false), coeff, rtol=1e-6) # Test that the reconstructed coeff matrix is ≈ as the one saved
    @test isapprox(Int64.(reconstruct_antisymmetric(lags_eff, fillwithnans=false)), lags)  # Test that the reconstructed lags matrix is ≈ as the one saved
end

@testset "Type tests" begin

    types = [Float16, Float32, Float64]
    N = 1024
    τ = 1024

    I = typeof(τ)

    for T in types
        x = range(T(0), T(1), length=N)
        A = [sin.(T(2π)*x);;sin.(T(7.3*2π)*x);;sin.(T(5.2*2π)*x);;sin.(T(11.3*2π)*x)]

        NT = T == Float64 ? Float32 : T #Downcast to Float32 since Metal does not support Float64, else leave the type as is


        @test eltype(A) == NT
        @test eltype(memcopy(A)) == NT

        ncorr = norm_correlogram(A, A, τ, 1024)
        @test eltype(ncorr) == NT

        ncorr_eff = norm_correlogram(A, τ, 1024)
        @test eltype(ncorr_eff) == NT

        coeff, lags = simplelags(ncorr, τ)
        @test eltype(coeff) == NT
        @test eltype(lags) == I

        coeff_eff, lags_eff = simplelags(ncorr_eff, τ)
        @test eltype(coeff_eff) == NT
        @test eltype(lags_eff) == I

        # Move to CPU
        coeff_eff = memcopy(coeff_eff)
        lags_eff = memcopy(lags_eff)
        @test eltype(reconstruct_symmetric(coeff_eff, fillwithnans=false)) == NT
        @test eltype(reconstruct_antisymmetric(lags_eff, fillwithnans=false)) == Float64
    end
end
