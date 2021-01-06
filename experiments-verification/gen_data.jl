using Random

function gensignalARX(; M1=2, M2=2, T=100)

    # Orders
    M = M1+1+M2

    # Parameters
    τ_true = 1e3
    θ_true = 2e-1 .*randn(M,)

    # Input frequency and amplitude
    ω = 1/(2*π)
    A = range(0.99, stop=1.00, length=T)

    # Observation array
    input = zeros(T,)
    output = zeros(T,)
    errors = zeros(T,)

    for k = 1:T
        
        # Input
        input[k] = A[k]*cos(ω*k)
        
        # Generate noise
        errors[k] = sqrt(inv(τ_true))*randn(1)[1]
    
        # Output
        if k < M
            output[k] = input[k] + errors[k]
        else
            # Update history vectors
            x_kmin1 = output[k-1:-1:k-M1]
            z_kmin1 = input[k-1:-1:k-M2]
            
            # Compute output
            output[k] = θ_true'*[x_kmin1; input[k]; z_kmin1] + errors[k]
        end
    end

    return input, output
end


function gensignalNARX(ϕ, θ_scale, τ_true; M1=2, M2=2, degree=1, T=100)

    # Orders
    M = (M1+1+M2)*degree + 1

    # Parameters
    θ_true = θ_scale .*(rand(M,) .- 0.5)

    # Input frequency and amplitude
    ω = 1/(2*π)
    A = range(0.1, stop=1.0, length=T)
    A = ones(T,) / 2.

    # Observation array
    input = zeros(T,)
    output = zeros(T,)
    errors = zeros(T,)

    for k = 1:T
        
        # Input
        input[k] = A[k]*cos(ω*k)
        
        # Generate noise
        errors[k] = sqrt(inv(τ_true))*randn(1)[1]
    
        # Output
        if k < M
            output[k] = input[k] + errors[k]
        else
            # Update history vectors
            x_kmin1 = output[k-1:-1:k-M1]
            z_kmin1 = input[k-1:-1:k-M2]
            
            # Compute output
            output[k] = θ_true'*ϕ([x_kmin1; input[k]; z_kmin1]) + errors[k]
        end
    end

    return input, output
end


function split_data(T, split_index; start_index=10)

    # Check inputs
    if split_index > T
        error("Split index "*string(split_index)*"must be lower than end time T="*string(T))
    end

    # Select training set
    ix_trn = Array{Int64,1}(start_index:split_index)

    # Select validation set
    ix_val = Array{Int64,1}(split_index+1:T)

    return ix_trn, ix_val
end