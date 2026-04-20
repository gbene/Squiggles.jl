"""
Generate synthetic data for running benchmarks on both Julia and Matlab
"""

using MAT
using Squiggles
using JLD2


I = [2^x for x in 2:10]

n_samples = [2^x for x in 4:10]


for i in I
    out_vec = []

    for n in n_samples
        signals = SignalMatrix(n, 1, i, 1:20, 1, -0.5:0.1:0.5)
        push!(out_vec, signals)
    end
    @save "data/$i.jld2" out_vec

    file = matopen("data/$i.mat", "w")
    write(file, "out_vec", out_vec)
    close(file)
end
