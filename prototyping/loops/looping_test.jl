function foo(A, B, lag)
    size_a = length(A)
    size_b = length(B)




    if lag > 0

        for j = 1:size_a-lag
            signal_index = lag+j
            println("$j, $signal_index")

            # println("$j, $signal_index")
            # @inbounds s = signal[signal_index]
            # @inbounds t = template[j]
            # dot_prod += s*t

        end
    else
        for j = abs(lag)+1:size_a
            signal_index = lag+j
            println("$j, $signal_index")
        end
    end
end




A = rand(4)
B = rand(4)


foo(A,B, -2)
