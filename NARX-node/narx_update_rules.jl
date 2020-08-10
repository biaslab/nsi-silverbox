import LinearAlgebra: I, Hermitian, tr, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType
include("util.jl")

export ruleVariationalNARXOutNPPPPP,
       ruleVariationalNARXIn1PNPPPP,
       ruleVariationalNARXIn2PPNPPP,
       ruleVariationalNARXIn3PPPNPP,
	   ruleVariationalNARXIn4PPPPNP,
	   ruleVariationalNARXIn5PPPPPN

order_out = Nothing
order_inp = Nothing

function defineOrder(M::Int64, N::Int64)
    global order_out, order_inp
    order_out = M
	order_inp = N
end

# Approximating point for Taylor exp
mθ_ = zeros(3,)

function f(θ::Array{Float64,1}, x, u, z)
	"Hard-coded nonlinearity (todo: g as an input argument to rule)"
	return θ[1:order_out]'*x + θ[order_out+1]*u + θ[order_out+2:end]'*z
end

function hardcoded_Jacobian(mθ::Array{Float64,1}, x, u, z)
	"Hard-coded gradient of nonlinearity (todo: g as an input argument to rule)"
	return Array([x; u; z])
end

function ruleVariationalNARXOutNPPPPP(marg_y :: Nothing,
									  marg_θ :: ProbabilityDistribution{Multivariate},
                                      marg_x :: ProbabilityDistribution{Multivariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mz = unsafeMean(marg_z)
	mu = unsafeMean(marg_u)
	mτ = unsafeMean(marg_τ)
	Vθ = unsafeCov(marg_θ)

	# Set order
	M = dims(marg_x)
	N = dims(marg_z)
	defineOrder(M,N)

	# Evaluate f at mθ
	fθ = f(mθ, mx, mu, mz)

	# Set outgoing message
	return Message(Univariate, GaussianMeanPrecision, m=fθ, w=mτ)
end

function ruleVariationalNARXIn1PNPPPP(marg_y :: ProbabilityDistribution{Univariate},
                                      marg_θ :: Nothing,
									  marg_x :: ProbabilityDistribution{Multivariate},
                                      marg_z :: ProbabilityDistribution{Multivariate},
                                      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

    # Extract moments of beliefs
	my = unsafeMean(marg_y)
	mx = unsafeMean(marg_x)
	mz = unsafeMean(marg_z)
	mu = unsafeMean(marg_u)
	mτ = unsafeMean(marg_τ)

	# Set order
	M = dims(marg_x)
	N = dims(marg_z)
	defineOrder(M,N)

	# Jacobian of f evaluated at mθ
	Jθ = hardcoded_Jacobian(mθ_, mx, mu, mz)

	# Update parameters
	Φ = mτ*Jθ*Jθ'
	ϕ = mτ*my*Jθ

	# Update approximating point
	global mθ_ = inv(Φ + 1e-8*Matrix{Float64}(I, size(Φ)))*ϕ

	# Set message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNARXIn2PPNPPP(marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: Nothing,
							  	      marg_z :: ProbabilityDistribution{Multivariate},
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

    return Message(vague(GaussianWeightedMeanPrecision, 2))
end

function ruleVariationalNARXIn3PPPNPP(marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
							  	      marg_z :: Nothing,
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

    return Message(vague(GaussianWeightedMeanPrecision, 2))
end

function ruleVariationalNARXIn4PPPPNP(marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
							  	      marg_z :: ProbabilityDistribution{Multivariate},
								      marg_u :: Nothing,
                                      marg_τ :: ProbabilityDistribution{Univariate})

    return Message(vague(GaussianWeightedMeanPrecision))
end

function ruleVariationalNARXIn5PPPPPN(marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
							  	      marg_z :: ProbabilityDistribution{Multivariate},
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_τ :: Nothing)

    # Extract moments of beliefs
	my = unsafeMean(marg_y)
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mz = unsafeMean(marg_z)
	mu = unsafeMean(marg_u)
	Vθ = unsafeCov(marg_θ)

	# Set order
	M = dims(marg_x)
	N = dims(marg_z)
	defineOrder(M,N)

	# Evaluate f at mθ
	fθ = f(mθ, mx, mu, mz)

	# Jacobian of f evaluated at mθ
	Jθ = hardcoded_Jacobian(mθ, mx, mu, mz)

	# Update parameters
	a = 3/2.
	b = (my^2 - 2*my*fθ + fθ^2 + Jθ'*Vθ*Jθ)/2.

	# Set message
    return Message(Univariate, Gamma, a=a, b=b)
end
