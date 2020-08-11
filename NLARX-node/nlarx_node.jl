using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov
import SpecialFunctions: polygamma, digamma
export NLatentAutoregressiveX, NLARX, averageEnergy, slug

"""
Description:

    A Nonlinear Latent Autoregressive model with eXogenous input (NLARX).

    The node function is a Gaussian with mean-precision parameterization:

    f(y, θ, x, η, u, γ) = 𝒩(y | A(θ)x + B(η)u, V(γ)),

    where A(θ) = | θ_1  …  θ_M | , B(η) = | η | ,
                 | I_M-1	 0 |          | 0 |

        for AR-order M.

Interfaces:

    1. y (output vector)
    2. θ (autoregression coefficients)
    3. x (input vector)
    4. η (control coefficients)
    5. u (control)
    6. γ (precision)

Construction:

    NLatentAutoregressiveX(y, θ, x, η, u, γ, id=:some_id)
"""

mutable struct NLatentAutoregressiveX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    g::Function # Scalar function between autoregression coefficients and state variable

    function NLatentAutoregressiveX(y, θ, x, η, u, γ; g::Function=x->x, id=generateId(NLatentAutoregressiveX))
        @ensureVariables(y, x, θ, η, u, γ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:θ] = self.interfaces[3] = associate!(Interface(self), θ)
        self.i[:η] = self.interfaces[4] = associate!(Interface(self), η)
        self.i[:u] = self.interfaces[5] = associate!(Interface(self), u)
        self.i[:γ] = self.interfaces[6] = associate!(Interface(self), γ)
        return self
    end
end

slug(::Type{NLatentAutoregressiveX}) = "NLARX"

function averageEnergy(::Type{NLatentAutoregressiveX},
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_η::ProbabilityDistribution{Univariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_γ::ProbabilityDistribution{Univariate})

    error("not implemented yet")

end
