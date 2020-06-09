# Explore Silverbox setting
# Euler's method and AR nodes
#
# Wouter Kouw
# 12-03-2020

using CSV
using DataFrames

# Read data from CSV file
df = CSV.read("../data/SNLS80mV.csv", ignoreemptylines=true)
df = select(df, [:V1, :V2])

# Shorthand
input = df[:,1]
output = df[:,2]

# Time horizon
T = size(df, 1);

"""
State-space formulation of Silverbox's dynamics:

μ⋅d^2/dt^2 x(t) + ν⋅d/dt x(t) + κ(x(t))⋅x(t) = u(t) + w(t)
κ(x(t)) = α + β x^2(t)
y(t) = x(t) + e(t)

where
μ       = mass
ν       = viscous damping
κ(x(t)) = nonlinear spring
y(t)    = observation (displacement)
x(t)    = state (displacement)
u(t)    = force
e(t)    = measurement noise
w(t)    = process noise

I now take a series of steps to re-write this problem:

1. Assume constant spring coefficient κ

μ⋅ d^2/dt^2 x(t) + ν⋅d/dt x(t) + κ⋅x(t) = u(t) + w(t)

2. Divide by leading coefficient

d^2/dt^2 x(t) + (ν/μ)⋅d/dt x(t) + (κ/μ)⋅ x(t) - u(t)/μ = w(t)/μ

3. Substitute variables (natural frequency ω_0 and damping ratio ζ)

d^2/dt^2 x(t) + 2ζω_0⋅d/dt x(t) + ω_0^2⋅x(t) - u(t)/μ = w(t)/μ

4. Apply Euler's method to obtain difference equation (step size is 1)

-> Forward Euler:

(x(t+2h)-2x(t+h)+x(t))/h^2 + 2ζω_0(x(t+h)-x(t))/h + ω_0^2⋅ x(t) - u(t)/μ = w(t)/μ
           x(t+2) + 2(ζω_0 - 1)x(t+1) + (1 - 2ζω_0 + ω_0^2)x(t) - u(t)/μ = w(t)/μ

-> Backward Euler:

(x(t)-2x(t-h)+x(t-2h))/h^2 + 2ζω_0(x(t)-x(t-h))/h + ω_0^2⋅ x(t) - u(t)/μ = w(t)/μ
           (1 + 2ζω_0 + ω_0^2)x(t) - 2(1 + ζω_0)x(t-1) - x(t-2) - u(t)/μ = w(t)/μ
        ↓
    x_t = α⋅x_t-1 + β⋅x_t-2 + γ⋅u_t = γ⋅w_t

    where α = 2(1 + ζω_0)/(1 + 2ζω_0 + ω_0^2),
          β = 1/(1 + 2ζω_0 + ω_0^2),
          γ = 1/(μ(1 + 2ζω_0 + ω_0^2))

6. Convert to multivariate first-order difference form

- Backward Euler:

    z_t = M⋅z_t-1 + N⋅u_t + N⋅w_t

    where z_t = [x_t x_t-1],
            M = [α β;
                 1 0],
            N = [γ 0]

5. Convert to Gaussian probability

- Backward Euler:

    z_t ~ Normal(M⋅z_t-1 + N⋅u_t, N⋅τ)

    where w_t ~ Normal(0, τ)

7. Observation likelihood

    y_t ~ Normal(H⋅z_t, σ)

    where e_t ~ Normal(0, σ), c = [1 0]

Now, I need priors for α, β, γ, τ, σ. Given three equations and three unknowns,
I can recover ζ, ω_0 and μ from α, β, and γ. The variables are all strictly
positive, which means they should be modeled by gamma distributions:

α ~ Γ(1, 1e3)
β ~ Γ(1, 1e3)
γ ~ Γ(1, 1e3)
τ ~ Γ(1, 1e3)
σ ~ Γ(1, 1e3)

I will introduce another shorthand: θ = [α β] and use the Nonlinear{Unscented}
node, with θ = exp(η), provide the AR node with a Gaussian form for θ.

"""

using ForneyLab
using LAR
using ForneyLab: unsafeMean
using LAR.Node, LAR.Data
using ProgressMeter

# Start graph
graph = FactorGraph()

# Static parameters
@RV η ~ GaussianMeanPrecision(placeholder(:η_m, dims=(2,)), placeholder(:η_w, dims=(2,2)))
@RV τ ~ Gamma(placeholder(:τ_a), placeholder(:τ_b))

# Nonlinear function
g(x) = exp.(x)
g_inv(x) = log.(x)

# Nonlinear node
# @RV θ ~ Nonlinear(log_θ; g=g, g_inv=g_inv, dims=(2,))
@RV θ ~ Nonlinear{Unscented}(η; g=g, dims=(2,))

# I'm fixing measurement noise σ
σ = 0.0001

# Observation selection variable
c = [1, 0]

# State prior
@RV z_t ~ GaussianMeanPrecision(placeholder(:z_m, dims=(2,)), placeholder(:z_w, dims=(2, 2)))

# Autoregressive node
@RV x_t ~ Autoregressive(θ, z_t, τ)

# Specify likelihood
@RV y_t ~ GaussianMeanVariance(dot(c, x_t), σ)

# Placeholder for data
placeholder(y_t, :y_t)

# Draw time-slice subgraph
ForneyLab.draw(graph)

# Infer an algorithm
q = PosteriorFactorization(z_t, x_t, θ, η, τ, ids=[:z, :x, :θ, :η, :τ])
algo = variationalAlgorithm(q, free_energy=true)
source_code = algorithmSourceCode(algo, free_energy=true)
eval(Meta.parse(source_code))

# Looking at only the first few timepoints
# T = 1000
T = size(df, 1);

# Inference parameters
num_iterations = 10

# Initialize marginal distribution and observed data dictionaries
data = Dict()
marginals = Dict()

# Initialize arrays of parameterizations
params_x = (zeros(2,T+1), repeat(1. .*float(eye(2)), outer=(1,1,T+1)))
params_z = (zeros(2,T+1), repeat(1. .*float(eye(2)), outer=(1,1,T+1)))
params_θ = (zeros(2,T+1), repeat(1. .*float(eye(2)), outer=(1,1,T+1)))
params_η = (zeros(2,T+1), repeat(1. .*float(eye(2)), outer=(1,1,T+1)))
params_τ = (1e-3.*ones(1,T+1), ones(1,T+1))

# Start progress bar
p = Progress(T, 1, "At time ")

# Perform inference at each time-step
for t = 1:T

    # Update progress bar
    update!(p, t)

    # Initialize marginals
    marginals[:x_t] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=params_x[1][:,t], w=params_x[2][:,:,t])
    marginals[:z_t] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=params_z[1][:,t], w=params_z[2][:,:,t])
    marginals[:η] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=params_η[1][:,t], w=params_η[2][:,:,t])
    marginals[:θ] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=params_θ[1][:,t], w=params_θ[2][:,:,t])
    marginals[:τ] = ProbabilityDistribution(Univariate, Gamma, a=params_τ[1][1,t], b=params_τ[2][1,t])

    data = Dict(:y_t => output[t],
                :θ_m => params_θ[1][:,t],
                :η_m => params_η[1][:,t],
                :z_m => params_z[1][:,t],
                :θ_w => params_θ[2][:,:,t],
                :η_w => params_η[2][:,:,t],
                :z_w => params_z[2][:,:,t],
                :τ_a => params_τ[1][1,t],
                :τ_b => params_τ[2][1,t])

    # Iterate variational parameter updates
    for i = 1:num_iterations

        stepx!(data, marginals)
        stepz!(data, marginals)
        stepθ!(data, marginals)
        stepη!(data, marginals)
        stepτ!(data, marginals)
    end

    # Store current parameterizations of marginals
    params_x[1][:,t+1] = unsafeMean(marginals[:x_t])
    params_z[1][:,t+1] = unsafeMean(marginals[:z_t])
    params_η[1][:,t+1] = unsafeMean(marginals[:η])
    params_θ[1][:,t+1] = unsafeMean(marginals[:θ])
    params_x[2][:,:,t+1] = marginals[:x_t].params[:w]
    params_z[2][:,:,t+1] = marginals[:z_t].params[:w]
    params_η[2][:,:,t+1] = marginals[:η].params[:w]
    params_θ[2][:,:,t+1] = marginals[:θ].params[:w]
    params_τ[1][1,t+1] = marginals[:τ].params[:a]
    params_τ[2][1,t+1] = marginals[:τ].params[:b]

end

# Estimates of process noise over time
println(params_τ[1] ./ params_τ[2])

# Extract estimated states
estimated_states = params_x[1][1,2:end]

"Visualize results"

using Plots

# Plot every n-th time-point to avoid figure size exploding
n = 10

p1 = Plots.scatter(1:n:T, output[1:n:T],
                   color="black",
                   label="output",
                   markersize=2,
                   size=(1600,800),
                   xlabel="time (t)",
                   ylabel="response")
Plots.plot!(1:n:T, estimated_states[1:n:T],
            color="red",
            linewidth=1,
            label="estimated")
Plots.savefig(p1, "viz/estimated_states01.png")
