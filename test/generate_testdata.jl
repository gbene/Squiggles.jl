using Squiggles
using CUDA
using Dates
using JLD2

set_GPUbackend()
N = 1024
x = range(0, 1, length=N)
A = [sin.(2π*x);;sin.(7.3*2π*x);;sin.(5.2*2π*x);;sin.(11.3*2π*x)]

ncorr = norm_correlogram(A, A, N, N)
ncorr_eff = norm_correlogram(A, N, N)

coeff, lags = simplelags(ncorr, N)
coeff_eff, lags_eff = simplelags(ncorr_eff, N)

save("$(@__DIR__)/files/$N.jld2","A", A, "coeff", memcopy(coeff), "lags", memcopy(lags), "date_created", now())
