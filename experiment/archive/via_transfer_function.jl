# Explore Silverbox setting
# Regression of transfer function
#
# Wouter Kouw
# 11-03-2020

using CSV
using DataFrames

# Read data from CSV file
df = CSV.read("data/SNLS80mV.csv", ignoreemptylines=true)
df = select(df, [:V1, :V2])

# Time horizon
T = size(df, 1)

"""
State-space formulation of Silverbox's dynamics:

μ⋅ d^2/dt^2 x(t) + ν⋅d/dt x(t) + κ(t)⋅ x(t) = u(t) + w(t)
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

We start with the noiseless differential equation and divide by the leading
coefficient, mass. We will ignore the spring coefficient function and just
model a constant coefficient κ. The control signal is considered a sequence
of impulses. This implies that it imposes a set of constraints on the impulse
response function of an unforced damped harmonic oscillator:

    d^2/dt^2 x(t) + (ν/μ)⋅ d/dt x(t) + (κ/μ)⋅ x(t) = 0

    with  x(0) = 0, x'(0) = 1 for t=0 until t=1.

The solution to this equation is of the form:

    x(t) = (exp(s_1⋅t) - exp(s_2⋅t)) / (s_2 - s_1)

where s_1, s_2 are the roots of the polynomial s^2 + (ν/μ)s + (κ/μ) = 0. The
denominator is in fact the transfer function (via the Laplace transform):

    G(s) = 1/(s_2 - s_1)

Perhaps we can regress the parameters of the transfer function.

"""
