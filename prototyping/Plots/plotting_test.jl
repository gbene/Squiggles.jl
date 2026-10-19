using GLMakie
using Squiggles
using LinearAlgebra
using Statistics

"""
    plotSignalMatrix(M)


Plot matrix of signals as a series of squiggles. The matrix should be i x n where i is the number of samples and n the number of signals
"""
function plotSignalMatrix(M::Matrix; sep_step=2, fig_display=false)

    fig = Figure()

    n_samples, n_signals = size(M)

    ax = Axis(fig[1,1], aspect=DataAspect(), title="Signal matrix", xlabel="Sample number",
              ylabel="Signal index", yticks=(1:sep_step:n_signals*sep_step, string.(1:n_signals)))



    for i in 1:n_signals

        signal = M[:,i]

        l = lines!(ax, signal)

        translate!(l, 0, (i-1)*sep_step)


    end

    if fig_display
        display(fig)
    end

    return fig, ax
end

# sampling_rate = 100
# duration = 1 #s
# n_signals = 10

# freq_range = 1:20
# phase_range = 0:0.1:duration
# padding = 3
# delay_range = -2:0.25:2

# M = zeros(Int(0.4*sampling_rate), n_signals)


# for i in 1:n_signals
#       delay = rand(delay_range)
#       event, domain = RandomEvent(freq_range, phase_range, [1], sampling_rate, duration, 10)
#       trace, time_trace = AddPadding(event, sampling_rate, padding, delay)

#       template, ttime = ExtractSnippet(trace, time_trace, sampling_rate, padding+delay+0.3,padding+delay+duration-0.3)
#       M[:,i] = real(template)
# end

M = SignalMatrix(128, 1, 10, 1:20, 1, -1:0.25:1)

plotSignalMatrix(M, sep_step=10; fig_display=true)
