using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov
import SpecialFunctions: polygamma, digamma
export GeneralisedFilterX, GFX, averageEnergy, slug

"""
Description:

    A node for transitions between states in generalised coordinates.

    1. A Gaussian mixture with mean-precision parameterization:

    𝒩(y | A(θ,x) + B(η,u), V(γ)),

    where A(x) = Sx + cg(x, θ), B(η,u) = c η u, and V(γ) = γI

    with S = | 1  …  Δt | ,  c = | 1 |
             |    …  Δt |        | . |
             | 0      1 |        | 0 |

    Interfaces:
        1. y (output vector)
        2. θ (state coefficients)
        3. x (generalised coordinates)
        4. η (control coefficients)
        5. u (exogenous input)
        6. γ (precision)

    Construction:
        GeneralisedFilterX(y, θ, x, η, u, γ, g=Function, id=:some_id)

    2. A deterministic state transition

        δ(y - (A(θ,x) + B(η,u))

        where A(x) = Sx + cg(x, θ) and B(η,u) = c η u

        with S = | 1  …  Δt | ,  c = | 1 |
                 |    …  Δt |        | . |
                 | 0      1 |        | 0 |

    Interfaces:
        1. y (output vector)
        2. θ (state coefficients)
        3. x (generalised coordinates)
        4. η (control coefficients)
        5. u (exogenous input)

    Construction:
        GeneralisedFilterX(y, θ, x, η, u, g=Function, id=:some_id)

"""

mutable struct GeneralisedFilterX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    g::Function # Scalar function between autoregression coefficients and state variable

    function GeneralisedFilterX(y, θ, x, η, u, γ; g::Function=x->x, Δt::Float=1., id=generateId(GeneralisedFilterX))
        @ensureVariables(y, x, θ, η, u, γ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}(), g)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:θ] = self.interfaces[3] = associate!(Interface(self), θ)
        self.i[:η] = self.interfaces[4] = associate!(Interface(self), η)
        self.i[:u] = self.interfaces[5] = associate!(Interface(self), u)
        self.i[:γ] = self.interfaces[6] = associate!(Interface(self), γ)
        return self
    end

    function GeneralisedFilterX(y, θ, x, η, u; g::Function=x->x, Δt::Float=1., id=generateId(GeneralisedFilterX))
        @ensureVariables(y, x, θ, η, u)
        self = new(id, Array{Interface}(undef, 5), Dict{Symbol,Interface}(), g)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:θ] = self.interfaces[3] = associate!(Interface(self), θ)
        self.i[:η] = self.interfaces[4] = associate!(Interface(self), η)
        self.i[:u] = self.interfaces[5] = associate!(Interface(self), u)
        return self
    end
end

slug(::Type{GeneralisedFilterX}) = "GFX"

function averageEnergy(::Type{GeneralisedFilterX},
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_η::ProbabilityDistribution{Univariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_γ::ProbabilityDistribution{Univariate})

    #TODO
    error("not implemented yet")
end
