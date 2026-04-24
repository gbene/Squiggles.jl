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


"""
    plotCC(cc_mat)

Plot the correlation coefficient matrix
"""
function plotCC(cc_mat)

    fig = Figure(size=(900,900))
    ax = Axis(fig[1, 1], yreversed=true, title="Correlation coefficient", ylabel="i", xlabel="j")
    heatmap!(ax, cc_mat)

    display(fig)

    return fig, ax

end


"""
    plotCorrelogram(correlograms)

Interactive plot of the correlogram volume.

"""
function plotCorrelogram(correlograms)

    perm_corrs = permutedims(correlograms,(3,1,2))
    n_signals, n_templates, n_lags = size(correlograms)
    τ = (n_lags-1)/2
    fig = Figure(size=(900,900))
    ax = Axis3(fig[1, 1], xlabel="Lags", ylabel="i", zlabel="j")

    sl_y = Slider(fig[2, 1], range = 1:n_signals, startvalue = 1)
    sl_z = Slider(fig[1, 2], range = 1:n_templates, startvalue = 1, horizontal=false)
    ax2 = Axis(fig[1,3], xlabel="Lags", ylabel="Correlation", title="Correlogram" )
    ylims!(ax2, -1, 1)

    points = lift(sl_y.value, sl_z.value) do y, z
        [Point3f(-τ*1.1, y+0.5, z+0.5), Point3f(τ*1.1, y+0.5, z+0.5)]
    end

    text_pos = lift(sl_y.value, sl_z.value) do y, z
        Point3f(-τ*1.1, y+0.5, z+0.5)
    end

    text_label = lift(sl_y.value, sl_z.value) do y, z
        "$y,$z"
    end

    correlogram = lift(sl_y.value, sl_z.value) do y, z
        perm_corrs[:,y,z]
    end

    lines!(ax, points, color = :black, )
    textlabel!(ax, text_pos, text=text_label, fontsize = 15)
    lines!(ax2,correlogram)


    volume!(ax, (-τ,τ), (1,n_signals),(1,n_templates), perm_corrs, interpolate = false)

    display(fig)

    return fig, ax

end
