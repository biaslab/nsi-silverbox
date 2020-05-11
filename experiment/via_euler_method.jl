# Explore Silverbox setting
# Euler's method and AR nodes
#
# Wouter Kouw
# 12-03-2020

using CSV
using DataFrames

# Read data from CSV file
df = CSV.read("data/SNLS80mV.csv", ignoreemptylines=true)
df = select(df, [:V1, :V2])

# Time horizon
T = size(df, 1)

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

"""

using ForneyLab
using LAR
using ProgressMeter

# Start graph
g = FactorGraph()

# Static parameters
@RV α ~ Gamma(1, 1e3)
@RV β ~ Gamma(1, 1e3)
@RV γ ~ Gamma(1, 1e3)
@RV τ ~ Gamma(1, 1e3)

# I'm fixing measurement noise σ
σ = 10.

# Observation selection variable
c = [1, 0]

# Convenience variables
θ = [α β]

# State prior
@RV z_t ~ GaussianMeanPrecision(placeholder(:m_z_t, dims=(2,)), placeholder(:w_z_t, dims=(2, 2)))

# Autoregressive node
@RV x_t ~ Autoregressive(θ, z_t, τ)

# Specify likelihood
@RV y_t ~ GaussianMeanPrecision(dot(c, x_t), σ)

# Placeholder for data
placeholder(y_t, :y_t)

# Recognition factorization
q = RecognitionFactorization(x_t, z_t, θ, τ, ids=[:x, :z, :θ, :τ])
algo = variationalAlgorithm(q)
eval(Meta.parse(algo))

# Inference parameters
num_iterations = 10

# Initialize marginal distribution and observed data dictionaries
data = Dict()
marginals = Dict()

# Initialize arrays of parameterizations
params_x = zeros(T+1,2)
params_z = zeros(T+1,2)
params_θ = zeros(T+1,2)
params_τ = zeros(T+1,2)

# Start progress bar
p = Progress(T, 1, "At time ")

# Perform inference at each time-step
for t = 1:T

    # Update progress bar
    update!(p, t)

    # Initialize marginals
    marginals[:x] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=params_x[t,1], w=params_x[t,2])
    marginals[:z] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=params_z[t,1], w=params_z[t,2])
    marginals[:θ] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=params_θ[t,1], w=params_θ[t,2])
    marginals[:τ] = ProbabilityDistribution(Univariate, Gamma, a=params_τ[1], b=params_τ[2])

    # Iterate variational parameter updates
    for i = 1:num_iterations

        stepx!(data, marginals)
        stepz!(data, marginals)
        stepθ!(data, marginals)
        stepτ!(data, marginals)
    end

    # Store current parameterizations of marginals
    params_x[t+1,1] = unSafeMean(marginals[:x])
    params_x[t+1,2] = unSafePrecision(marginals[:x])
    params_z[t+1,1] = unSafeMean(marginals[:z])
    params_z[t+1,2] = unSafePrecision(marginals[:z])
    params_θ[t+1,1] = unSafeMean(marginals[:θ])
    params_θ[t+1,2] = unSafePrecision(marginals[:θ])
    params_τ[t+1,1] = marginals[:τ].params[:a]
    params_τ[t+1,2] = marginals[:τ].params[:b]

end

end
