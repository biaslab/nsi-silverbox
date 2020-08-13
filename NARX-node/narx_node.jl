using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov
export NAutoregressiveX, NARX

"""
Description:

    A Nonlinear Autoregressive model with eXogeneous input (NARX)

    y_t = f(y_t-1, …, y_t-M, u_t, u_t-1, …, u_t-N) + e_t

    where M is the auto-regression order of observations and N of inputs.
    Concatenate previous observations: x_t-1 = [y_t-1, …, y_t-M]' and
    previous inputs z_t-1 = [u_t-1, …, u_t-N]'.

    Assume y_t, x_t-1, z_t-1 and u_t are observed and e_t ~ N(0, τ^-1).

Interfaces:

    1. y (output)
    2. θ (function coefficients)
    3. x (previous observations vector)
    4. z (previous inputs vector)
    5. u (input)
    6. τ (precision)

Construction:

    NAutoregressiveX(y, θ, x, z, u, τ, g=f, id=:some_id)
"""

mutable struct NAutoregressiveX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    g::Function # Scalar function between autoregression coefficients and state variable

    function NAutoregressiveX(y, θ, x, z, u, τ; g::Function, id=generateId(NAutoregressiveX))
        @ensureVariables(y, θ, x, z, u, τ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}(), g)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:θ] = self.interfaces[2] = associate!(Interface(self), θ)
        self.i[:x] = self.interfaces[3] = associate!(Interface(self), x)
        self.i[:z] = self.interfaces[4] = associate!(Interface(self), z)
        self.i[:u] = self.interfaces[5] = associate!(Interface(self), u)
        self.i[:τ] = self.interfaces[6] = associate!(Interface(self), τ)
        return self
    end
end

slug(::Type{NAutoregressiveX}) = "NARX"

function averageEnergy(::Type{NAutoregressiveX},
                       marg_y::ProbabilityDistribution{Univariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_z::ProbabilityDistribution{Multivariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_τ::ProbabilityDistribution{Univariate})

    error("not implemented yet")

end
