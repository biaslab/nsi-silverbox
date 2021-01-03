using Random

function gensignalARX(; M1=2, M2=2, T=100)

    # Orders
    M = M1+1+M2

    # Parameters
    τ_true = 1e3
    θ_true = 2e-1 .*randn(M,)

    # Basis function
    ϕ(x) = x

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
            output[k] = θ_true'*ϕ([x_kmin1; input[k]; z_kmin1]) + errors[k]
        end
    end

    return output, input, ϕ, (θ_true, τ_true)
end


function gensignalNARX(; M1=2, M2=2, deg=1, T=100)

    # Orders
    M = M1+1+M2
    N = (M1+1+M2)*deg + 1

    # Parameters
    τ_true = 1e3
    θ_true = 2e-1 .*randn(N,)

    # Nonlinearity
    C = zeros(M,1); for d=1:deg; C = hcat(d .*Matrix{Float64}(I,M,M), C); end
    ϕ(x::Array{Float64,1}) = [prod(x.^C[:,k]) for k = 1:size(C,2)]

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
            output[k] = θ_true'*ϕ([x_kmin1; input[k]; z_kmin1]) + errors[k]
        end
    end

    return output, input, ϕ, (θ_true, τ_true)
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