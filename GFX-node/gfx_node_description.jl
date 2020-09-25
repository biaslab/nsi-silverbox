using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, differentialEntropy, Interface, Variable, slug, ProbabilityDistribution,
				  unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov,
				  ultimatePartner, posteriorFactor, assembleClamp!
export GeneralisedFilterX, GFX

"""
Description:

    A node for transitions between states in generalised coordinates.

    1. A Gaussian mixture with mean-precision parameterization:

    𝒩(y | A(θ)x + B(η)u, V(γ)),

    where A(x) = (S + cθ')x, B(η) = ηc, and

    with S = | 1  …  Δt | ,  c = | 0 | ,  V(γ) = | ϵ  …  … 0  |
             | .  …  Δt |        | . |           | 0  ϵ  … 0  |
             | .  …   . |        | . |           | .  .  … 0  |
             | 0  …   1 |        | 1 |           | 0  …  … γ⁻¹|

    Interfaces:
        1. y (output vector)
        2. θ (state coefficients)
        3. x (generalised coordinates)
        4. η (control coefficients)
        5. u (exogenous input)
        6. γ (precision)

    Construction:
        GeneralisedFilterX(y, θ, x, η, u, γ, Δt=1., id=:some_id)

    2. A deterministic state transition

        δ(y - (A(θ)x + B(η)u))

        where A(x) = (S + cθ')x, B(η) = ηc, and

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
        GeneralisedFilterX(y, θ, x, η, u, Δt=1., id=:some_id)

"""

mutable struct GeneralisedFilterX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    # Sampling time
    Δt::Float64

    function GeneralisedFilterX(y, θ, x, η, u, γ; Δt::Float64=1., id=generateId(GeneralisedFilterX))
        @ensureVariables(y, x, θ, η, u, γ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}(), Δt)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:θ] = self.interfaces[2] = associate!(Interface(self), θ)
        self.i[:x] = self.interfaces[3] = associate!(Interface(self), x)
        self.i[:η] = self.interfaces[4] = associate!(Interface(self), η)
        self.i[:u] = self.interfaces[5] = associate!(Interface(self), u)
        self.i[:γ] = self.interfaces[6] = associate!(Interface(self), γ)
        return self
    end

    function GeneralisedFilterX(y, θ, x, η, u; Δt::Float64=1., id=generateId(GeneralisedFilterX))
        @ensureVariables(y, x, θ, η, u)
        self = new(id, Array{Interface}(undef, 5), Dict{Symbol,Interface}(), Δt)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:θ] = self.interfaces[2] = associate!(Interface(self), θ)
        self.i[:x] = self.interfaces[3] = associate!(Interface(self), x)
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

function averageEnergy(::Type{GeneralisedFilterX},
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_η::ProbabilityDistribution{Univariate},
                       marg_u::ProbabilityDistribution{Univariate})

    #TODO
    error("not implemented yet")
end