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
