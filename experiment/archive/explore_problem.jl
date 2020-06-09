# Explore Silverbox setting
#
# Wouter Kouw
# 17-12-2019

using CSV
using DataFrames

# Read data from CSV file
df = CSV.read("data/SNLS80mV.csv", ignoreemptylines=true)
df = select(df, [:V1, :V2])

# Time horizon
T = size(df, 1)

"""
State-space formulation of Silverbox's dynamics:

d^2 x(t) / dt^2 ⋅ μ + dx(t) / dt ⋅ ν + κ(t) ⋅ x(t) = u(t) + w(t)
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

I translate this into the following second-order difference equation:

(x_t - 2x_t-1 + x_t-2)μ + (x_t - x_t-1)ν + κ_t x_t - u_t = w_t
y_t = x_t + e_t

where e_t ~ N(0, σ_e), w_t ~ N(0, σ_w). Note that I absorbed α + β x^2(t)
into κ_t.

If I divide by the leading coefficient, move everything to the right-handside
and integrate out process noise w_t, I produce the following distributions:

x_t ~ N( (2μ + ν)/(μ + ν + κ_t) x_t-1 - μ/(μ + ν + κ_t) x_t-2 + u_t, σ_w)
y_t ~ N(x_t, σ_e)

Note that I absorbed the leading coefficient w_t. Now, I need priors for μ, ν,
u_t, κ_t, σ_w, σ_e and x_0. All of these are strictly positive variables, which
means they should be modeled by gamma distributions:

μ   ~ Γ(1, 1e3)
ν   ~ Γ(1, 1e3)
u_t ~ Γ(1, 1e3)
σ_w ~ Γ(1, 1e3)
σ_e ~ Γ(1, 1e3)

"""

using ForneyLab

g = FactorGraph()

@RV μ ~ Gamma(1, 1e3)
@RV ν ~ Gamma(1, 1e3)
@RV σ_w ~ Gamma(1, 1e3)
@RV σ_e ~ Gamma(1, 1e3)
@RV x_0 ~ GaussianMeanVariance(0, 1e2)

κ = Vector{Variable}(undef, T)
u = Vector{Variable}(undef, T)
x = Vector{Variable}(undef, T)
y = Vector{Variable}(undef, T)
x_tmin1 = x_0
x_tmin2 = x_0
for t = 1:T

    @RV κ[t] ~ GaussianMeanVariance(0, 1)
    @RV x[t] ~ GaussianMeanVariance((2*μ + ν)/(μ + ν + κ[t])*x_tmin1 - μ/(μ + ν + κ[t])*x_tmin2 + u[t], σ_w)
    @RV y[t] ~ GaussianMeanVariance(x[t], σ_e)

    placeholder(u[t], :u, index=t)
    placeholder(y[t], :y, index=t)

    x_tmin1 = x[t]
    x_tmin2 = x_tmin1
end

# Define recognition factorization
RecognitionFactorization()

q_μ = RecognitionFactor(μ)
q_ν = RecognitionFactor(ν)
q_e = RecognitionFactor(σ_e)
q_w = RecognitionFactor(σ_w)
q_x_0 = RecognitionFactor(x_0)

q_κ = Vector{RecognitionFactor}(undef, T)
q_x = Vector{RecognitionFactor}(undef, T)
for t = 1:T
    q_κ[t] = RecognitionFactor(κ[t])
    q_x[t] = RecognitionFactor(x[t])
end

# Compile algorithm
algo_mf = variationalAlgorithm([q_μ; q_ν; q_e; q_w; q_x_0; q_x], name="MF")
algo_F_mf = freeEnergyAlgorithm(name="MF");

# Load algorithm
eval(Meta.parse(algo_mf))
eval(Meta.parse(algo_F_mf))

# Initialize data
data = Dict(:u => df[:V1], :y => df[:V2])
num_its = 10

# Initial recognition distributions
marginals_mf = Dict{Symbol, ProbabilityDistribution}(:μ => vague(Gamma),
                                                     :ν => vague(Gamma),
                                                     :σ_w => vague(Gamma),
                                                     :σ_e => vague(Gamma),
                                                     :x_0 => vague(GaussianMeanVariance))
for t = 1:T
    marginals_mf[:κ_*t] = vague(GaussianMeanVariance)
    marginals_mf[:x_*t] = vague(GaussianMeanVariance)
end

# Run algorithm
F_mf = Vector{Float64}(undef, num_its)
for i = 1:num_its
    stepMF!(data, marginals_mf)

    F_mf[i] = freeEnergyMF(data, marginals_mf)
end
