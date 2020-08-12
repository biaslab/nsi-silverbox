import LinearAlgebra: I, Hermitian, tr, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType,
	collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner
using Zygote
include("util.jl")

export ruleVariationalNARXOutNPPPPP,
       ruleVariationalNARXIn1PNPPPP,
       ruleVariationalNARXIn2PPNPPP,
       ruleVariationalNARXIn3PPPNPP,
	   ruleVariationalNARXIn4PPPPNP,
	   ruleVariationalNARXIn5PPPPPN

# Autoregression orders
order_out = Nothing
order_inp = Nothing

# Approximating point for Taylor series
approxθ = Nothing

function defineOrder(M::Int64, N::Int64)
    global order_out, order_inp, approxθ
    order_out = M
	order_inp = N
	approxθ = zeros(M+N+1,)
end


function ruleVariationalNARXOutNPPPPP(g :: Function,
									  marg_y :: Nothing,
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
	fθ = g(mθ, mx, mu, mz)

	# Set outgoing message
	return Message(Univariate, GaussianMeanPrecision, m=fθ, w=mτ)
end

function ruleVariationalNARXIn1PNPPPP(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
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
	# Jθ = Jacobian(g, [mθ_, mx, mu, mz])
	Jθ = Zygote.gradient(g, approxθ, mx, mu, mz)[1]

	# Update parameters
	Φ = mτ*Jθ*Jθ'
	ϕ = mτ*my*Jθ

	# Update approximating point
	global approxθ = inv(Φ + 1e-8*Matrix{Float64}(I, size(Φ)))*ϕ

	# Set message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNARXIn2PPNPPP(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: Nothing,
							  	      marg_z :: ProbabilityDistribution{Multivariate},
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

    return Message(vague(GaussianWeightedMeanPrecision, 2))
end

function ruleVariationalNARXIn3PPPNPP(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
							  	      marg_z :: Nothing,
								      marg_u :: ProbabilityDistribution{Univariate},
                                      marg_τ :: ProbabilityDistribution{Univariate})

    return Message(vague(GaussianWeightedMeanPrecision, 2))
end

function ruleVariationalNARXIn4PPPPNP(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
									  marg_θ :: ProbabilityDistribution{Multivariate},
									  marg_x :: ProbabilityDistribution{Multivariate},
							  	      marg_z :: ProbabilityDistribution{Multivariate},
								      marg_u :: Nothing,
                                      marg_τ :: ProbabilityDistribution{Univariate})

    return Message(vague(GaussianWeightedMeanPrecision))
end

function ruleVariationalNARXIn5PPPPPN(g :: Function,
									  marg_y :: ProbabilityDistribution{Univariate},
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
	fθ = g(mθ, mx, mu, mz)

	# Gradient of f evaluated at mθ
	Jθ = Zygote.gradient(g, mθ, mx, mu, mz)[1]

	# Update parameters
	a = 3/2.
	b = (my^2 - 2*my*fθ + fθ^2 + Jθ'*Vθ*Jθ)/2.

	# Set message
    return Message(Univariate, Gamma, a=a, b=b)
end


function collectNaiveVariationalNodeInbounds(node::NAutoregressiveX, entry::ScheduleEntry)
	inbounds = Any[]

	# Push function (and inverse) to calling signature
	# These functions needs to be defined in the scope of the user
	push!(inbounds, Dict{Symbol, Any}(:g => node.g,
									  :keyword => false))

    target_to_marginal_entry = currentInferenceAlgorithm().target_to_marginal_entry

    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface
            # Ignore marginal of outbound edge
            push!(inbounds, nothing)
        elseif (inbound_interface != nothing) && isa(inbound_interface.node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        else
            # Collect entry from marginal schedule
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        end
    end

    return inbounds
end
