using Revise
using JLD
using ProgressMeter
using LinearAlgebra
using ForneyLab
using NARX

import Base: invokelatest
import ForneyLab: unsafeMean, unsafeCov

include("gen_data.jl")
include("visualization.jl")


function generate_data(ϕ; deg=1, M1=1, M2=1, T=100, split_index=50, start_index=10)

    # Generate signal
    if deg == 1
        input, output = gensignalARX(M1=M1, M2=M2, T=T)
    else
        input, output = gensignalNARX(ϕ, M1=M1, M2=M2, deg=deg, T=T)
    end

    # Split data
    ix_trn, ix_val = split_data(T, split_index, start_index=start_index)

    return input, output, ix_trn, ix_val
end

function model_specification(ϕ; M1=M1, M2=M2, M=M)

    graph = FactorGraph()

    # Observed variables
    @RV x_kmin1; placeholder(x_kmin1, :x_kmin1, dims=(M1,))
    @RV z_kmin1; placeholder(z_kmin1, :z_kmin1, dims=(M2,))
    @RV u_k; placeholder(u_k, :u_k)

    # Time-invariant parameters
    @RV θ ~ GaussianMeanVariance(placeholder(:m_θ, dims=(M,)), placeholder(:v_θ, dims=(M,M)))
    @RV τ ~ Gamma(placeholder(:a_τ), placeholder(:b_τ))

    # Likelihood
    @RV y_k ~ NAutoregressiveX(θ, x_kmin1, u_k, z_kmin1, τ, g=ϕ)
    placeholder(y_k, :y_k)

    q = PosteriorFactorization(θ, τ, ids=[:θ :τ])
    algorithm = messagePassingAlgorithm([θ; τ], q)
    return algorithmSourceCode(algorithm)
    
end

function experiment_FEM(input, output, ix_trn, ix_val, ϕ; M1=1, M2=1, M=3, T=100, vis=false)

    T_trn = length(ix_trn)
    T_val = length(ix_val)

    "Inference execution"

    # Preallocate parameter arrays
    params_θ = (zeros(T_trn,M), zeros(T_trn,M,M))
    params_τ = (zeros(T_trn,1), zeros(T_trn,1))

    # Initialize priors
    θ_k = (zeros(M,), 5 .*Matrix{Float64}(I,M,M))
    τ_k = (1e3, 1e0)

    # Initialize marginals
    marginals = Dict(:θ => ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=θ_k[1], v=θ_k[2]),
                     :τ => ProbabilityDistribution(Univariate, Gamma, a=τ_k[1], b=τ_k[2]))

    # Initialize prediction arrays
    predictions = (zeros(T,), zeros(T,))

    for k in ix_trn
        
        # Update history vectors
        x_kmin1 = output[k-1:-1:k-M1]
        z_kmin1 = input[k-1:-1:k-M2]
        
        ϕx = ϕ([x_kmin1; input[k]; z_kmin1])
        predictions[1][k] = θ_k[1]'*ϕx
        predictions[2][k] = ϕx'*θ_k[2]*ϕx + inv(τ_k[1] / τ_k[2])
        
        # Set data 
        data = Dict(:y_k => output[k],
                    :u_k => input[k],
                    :x_kmin1 => x_kmin1,
                    :z_kmin1 => z_kmin1,
                    :m_θ => θ_k[1],
                    :v_θ => θ_k[2],
                    :a_τ => τ_k[1],
                    :b_τ => τ_k[2])
        
        # Iterate updates
        for n = 1:10
            stepθ!(data, marginals)
            stepτ!(data, marginals)
        end    
        
        # Update params
        θ_k = (unsafeMean(marginals[:θ]), unsafeCov(marginals[:θ]))
        τ_k = (marginals[:τ].params[:a], marginals[:τ].params[:b])
        
        # Store params
        params_θ[1][k-ix_trn[1]+1,:] = θ_k[1]
        params_θ[2][k-ix_trn[1]+1,:,:] = θ_k[2]
        params_τ[1][k-ix_trn[1]+1] = τ_k[1]
        params_τ[2][k-ix_trn[1]+1] = τ_k[2]
        
    end

    "Simulation"

    for k in ix_val
        
        if k == ix_val[1]
            # Update history vectors
            x_kmin1 = output[k-1:-1:k-M1]
            z_kmin1 = input[k-1:-1:k-M2]
        else
            # Update history vectors
            x_kmin1 = predictions[1][k-1:-1:k-M1]
            z_kmin1 = input[k-1:-1:k-M2]
        end
            
        # Posterior predictive
        ϕx = ϕ([x_kmin1; input[k]; z_kmin1])
        predictions[1][k] = θ_k[1]'*ϕx
        predictions[2][k] = ϕx'*θ_k[2]*ϕx + inv(τ_k[1] / τ_k[2])
        
    end

    "Evaluation"

    # Compute prediction errors
    pred_errors = (predictions[1][ix_val] - output[ix_val]).^2

    # Compute root mean square
    RMS = sqrt(mean(pred_errors))    

    if vis
        p1 = plot_forecast(output, predictions, ix_trn, ix_val, posterior=true)
        p2 = plot_errors(output, predictions[1], ix_val)

        savefig(p1, "figures/NARX-FEM_forecast.png")
        savefig(p2, "figures/NARX-FEM_errors.png")
    end

    return RMS, predictions
end

# Orders
M1 = 2
M2 = 2
deg = 1

# Signal lenghts
start_index = 10
split_index = 800 + start_index
time_horizon = 1600 + start_index

# Basis function
if deg == 1
    M = M1 + 1 + M2
    ϕ(x::Array{Float64,1}) = x
else
    M = (M1 + 1 + M2)*deg + 1
    global PP = zeros(M,1); 
    for d=1:deg; PP = hcat(d .*Matrix{Float64}(I,M,M), PP); end
    ϕ(x::Array{Float64,1}) = [prod(x.^PP[:,k]) for k = 1:size(PP,2)]
end

# Repetitions
num_repeats = 100
results = zeros(num_repeats,)
preds = zeros(time_horizon, num_repeats)

# Specify model and compile update functions
source_code = model_specification(ϕ, M1=M1, M2=M2, M=M)
eval(Meta.parse(source_code))

@showprogress for r = 1:num_repeats
    
    input, output, ix_trn, ix_val = generate_data(ϕ, deg=deg, M1=M1, M2=M2, T=time_horizon, split_index=split_index, start_index=start_index)
    results[r], predictions = experiment_FEM(input, output, ix_trn, ix_val, ϕ, M1=M1, M2=M2, M=M, T=time_horizon, vis=false)
    preds[:,r] = predictions[1]

end

# Report
println("Mean RMS = "*string(mean(results)))

# Write results to file
save("results/results_FEM_M"*string(M)*"_deg"*string(deg)*"_S"*string(split_index-start_index)*".jld", "RMS", results)