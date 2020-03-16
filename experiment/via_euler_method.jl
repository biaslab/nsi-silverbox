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

μ⋅d^2/dt^2 x(t) + ν⋅d/dt x(t) + κ(t)⋅x(t) = u(t) + w(t)
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

μ⋅ d^2/dt^2 x(t) + ν⋅d/dt x(t) + κ⋅ x(t) = u(t) + w(t)

2. Divide by leading coefficient

d^2/dt^2 x(t) + (ν/μ)⋅d/dt x(t) + (κ/μ)⋅ x(t) - u(t)/μ = w(t)/μ

3. Substitute variables (natural frequency ω_0 and damping ratio ζ)

d^2/dt^2 x(t) + 2ζω_0⋅d/dt x(t) + ω_0^2⋅ x(t) - u(t)/μ = w(t)/μ

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

    where z_t = [x_t, x_t-1], M = [α β; 1 0], N = [γ 0]

5. Convert to Gaussian probability

- Backward Euler:

    z_t ~ N(M⋅z_t-1 + N⋅u_t, N⋅τ)

    where w_t ~ N(0, τ)

7. Observation likelihood

    y_t ~ N(H⋅z_t, σ)

    where e_t ~ N(0, σ), c = [1 0]

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
using LAR.AR, LAR.DataAR
import LAR: logPDF, wmse

g = FactorGraph()

# Static parameters
@RV α ~ Gamma(1, 1e3)
@RV β ~ Gamma(1, 1e3)
@RV γ ~ Gamma(1, 1e3)
@RV τ ~ Gamma(1, 1e3)
@RV σ ~ Gamma(1, 1e3)

# Dynamic parameters
u = Vector{Variable}(undef, T)
x = Vector{Variable}(undef, T)
y = Vector{Variable}(undef, T)

x_tmin1 = x_0
x_tmin2 = x_0
for t = 1:T

    # Specify state transition
    @RV x[t] ~ GaussianMeanVariance((2*μ + ν)/(μ + ν + κ[t])*x_tmin1 - μ/(μ + ν + κ[t])*x_tmin2 + u[t], σ_w)

    # Specify likelihood
    @RV y[t] ~ GaussianMeanVariance(x[t], σ_e)

    # Specfify observed variables
    placeholder(u[t], :u, index=t)
    placeholder(y[t], :y, index=t)

    x_tmin1 = x[t]
    x_tmin2 = x_tmin1
end
