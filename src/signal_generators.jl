function Gauss(x, a, b, c)
        return a*exp(-((x-b)^2)/(2*c^2))
end

function Hamming(samp_freq)
    x = LinRange(-0.5,0.5, samp_freq)
    return 0.54 .+ 0.46*cos.(2π*x)
end

function Wave(x, amplitude, frequency, phase_shift)
    return amplitude*exp(im*(frequency*2π*(x+phase_shift)))
end

function _IterateSignalSum(signal,
                            freq_range, phase_range,
                            intensity_range,
                            duration, n_iterations)

    for i in 1:n_iterations
            freq = rand(freq_range)
            phase = rand(phase_range)
            amp = rand(intensity_range)
            signal += Wave.(duration, amp, freq, phase)#.*Hamming(length(duration))
    end
    return signal
end

"""
RandomEvent(freq_range, phase_range, intensity_range, samp_freq, duration, n_iterations)

Create a random phase arrival signal of a given duration and sampling rate. The signal is created by summing signals with random frequencies and phases.

# Arguments

- `freq_range`: Frequency or range of frequencies (in [Hz]) of the signals that are summed to create the final signal. If it is a range a frequency is randomly picked from the range.
- `phase_range`: Phase or range of phases (in [deg]) of the signals that are summed to create the final signal. If it is a range a frequency is randomly picked from the range.
- `intensity_range`: inentsity or range of intensities (between [0,1]) of the signals that are summed to create the final signal. If it is a range a value is randomly picked from the range.
- `samp_freq`: Sampling frequency (in [Hz]) of the signal
- `duration`: Duration of the signal (in [s])
- `n_iterations`: Number of summed signals. For each iteration a new random frequency/phase is picked depending on the freq_range/phase_range.

# Return

+ `x`: The final signal
+ `time`: The time from 0 to `duration`

# Examples

Create a 1s signal of 500Hz composed of 10 random signals with frequencies between 5 and 10Hz, phases between 1 and 180° and all with the same intensities (1)

```julia
event, etime = Signals.RandomEvent(1:5,1:180, [1], 500, 1, 3)
```


"""
function RandomEvent(freq_range, phase_range, intensity_range,
                    samp_freq, duration, n_iterations)


    n_samples = Int(samp_freq*duration)

    time = LinRange(0, duration, n_samples)


    x = zeros(ComplexF64, n_samples)


    x = _IterateSignalSum(x, freq_range, phase_range, intensity_range, time, n_iterations)


    return x, time
end

"""
AddPadding(signal, s_of_padding, s_of_delay)

Add 0 valued padding symmetrically to the signal. It is also possible to shift the input signal by a given amount .

# Arguments

- `signal`: Signal to which padding should be added
- `s_of_padding`: Amount of seconds to add to the signal
- `s_of_delay`: Amount of seconds to shift the signal

# Returns

- `padded_signal`: The signal with the added padding
- `time`: The time (in [s]) of the padded signal (starting from 0s)

# Examples

Create a 3s symmetrical buffer and delay the input by 2s

```julia
event, etime = Signals.RandomEvent(1:5,1:180, [1], 500, 1, 3)
padded_trace, time = AddPadding(event, 3, 2)
```

"""
function AddPadding(signal, samp_freq, s_of_padding, s_of_delay)

    n_samples = length(signal)
    signal_time = n_samples/samp_freq


    padding = Int(s_of_padding*samp_freq)
    padded_signal = zeros(ComplexF64, n_samples+2*padding)

    delay = Int(floor(s_of_delay*samp_freq))
    start_index = padding+delay

    end_index = start_index+n_samples
    padded_signal[start_index:end_index-1] = signal



    new_endtime = 2*s_of_padding+signal_time
    time = LinRange(0, new_endtime, n_samples+2*padding)

    center = s_of_padding+s_of_delay+(signal_time*0.5)
    # println(center)
    g = Gauss.(time, 1, center, 0.05)
    # println(center)
    padded_signal .*= g

    return padded_signal, time

end

function ExtractSnippet(signal, time, samp_freq, start_time_s, end_time_s)

    start_time_samp = Int(round(start_time_s*samp_freq))
    end_time_samp = Int(round(end_time_s*samp_freq))


    snippet = signal[start_time_samp:end_time_samp-1]

    time_snippet = time[start_time_samp:end_time_samp-1]

    return snippet, time_snippet

end

"""
AddNoise(signal, noise_range, noise_phase_range, intensity_range, n_iterations)

Add random noise to the input trace. The final noisy signal is created by summing signals with random frequencies and phases.

# Arguments

- `signal`: Signal to which padding should be added
- `noise_range`: Frequency or range of frequencies (in [Hz]) of the signals that are summed to create the final noise. If it is a range a frequency is randomly picked from the range.
- `noise_phase_range`: Phase or range of phases (in [deg]) of the signals that are summed to create the final noise. If it is a range a frequency is randomly picked from the range.
- `intensity_range`: inentsity or range of intensities (between [0,1]) of the signals that are summed to create the final signal. If it is a range a value is randomly picked from the range.
- `n_iterations`: Number of summed signals. For each iteration a new random frequency/phase is picked depending on the freq_range/phase_range.

# Return

+ `x`: The final noisy signal


# Examples

Add noise to a clean signal by adding 20 random signals with frequencies between 15 and 45Hz, phases between 1 and 180° and all scaled to 20%.

```julia-repl
event, etime = Signals.RandomEvent(1:5,1:180, [1], 500, 1, 3)
padded_trace, time = AddPadding(event, 3, 2)
noisy_trace = AddNoise(padded_trace, 15:45, 1:180, [0.2], 20)
```


"""
function AddNoise(signal, samp_freq,
                noise_range, noise_phase_range,
                intensity_range, n_iterations)

    n_samples = length(signal)
    duration = n_samples/samp_freq
    time = LinRange(0, duration, n_samples)

    signal = _IterateSignalSum(signal, noise_range, noise_phase_range, intensity_range, time, n_iterations)

    return signal
end


"""
    SignalMatrix(sampling_rate, duration, n_signals, freq_range, padding, delay_range)

Create a matrix of random signals. The matrix will have size sampling_rate x n_signals (rows x cols)

### Arguments

    -`sampling_rate::Int` -- Sampling rate of each signal
    -`duration::Number` -- Duration (in s) of each signal
    -`n_signals::Int` -- Number of signals
    -`freq_range::Range` -- Frequency range used to form the signal
    -`padding::Number` -- Amount of padding (in s) to add to the signals
    -`delay_range::Range` -- Delay to add to each signal

### Output

Real valued signal matrix

### Example

`SignalMatrix(100, 1, 10, 1:20, 3, -2:0.25:2)`

"""
function SignalMatrix(sampling_rate, duration, n_signals, freq_range, padding, delay_range)

    # sampling_rate = 100
    # duration = 1 #s
    # n_signals = 10

    # freq_range = 1:20
    phase_range = 0:0.1:duration
    # padding = 3
    # delay_range = -2:0.25:2

    M = zeros(Float32, sampling_rate, n_signals)

    for i in 1:n_signals
        # delay = rand(delay_range)
        event, domain = RandomEvent(freq_range, phase_range, [1], sampling_rate, duration, 10)
        # trace, time_trace = AddPadding(event, sampling_rate, padding, delay)

        # template, ttime = ExtractSnippet(trace, time_trace, sampling_rate, padding+delay,padding+delay+duration)
        M[:,i] = real(event)
    end

    return M
end
