import LinearAlgebra: I, Bidiagonal, tr
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType
				  collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner
import Zygote: gradient
include("util.jl")

export ruleVariationalGFXOutNPPPPP,
       ruleVariationalGFXIn1PNPPPP,
       ruleVariationalGFXIn2PPNPPP,
       ruleVariationalGFXIn3PPPNPP,
	   ruleVariationalGFXIn4PPPPNP,
	   ruleVariationalGFXIn5PPPPPN


function ruleVariationalGFXOutNPPPPP(Δt :: Float,
									 marg_y :: Nothing,
									 marg_θ :: ProbabilityDistribution{Multivariate},
                                     marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_η :: ProbabilityDistribution{Univariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mx,Vx = unsafeMeanCov(marg_x)
	mη,Vη = unsafeMeanCov(marg_η)
	mu,Vu = unsafeMeanCov(marg_u)
	mτ,Vτ = unsafeMeanCov(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = 1.

	# Cast precision to matrix
	mW = wMatrix(mτ, order)

	# Compute expected values
	EA = S + s*mθ'
	EB = s*mη

	# Set parameters
	ϕ = EA*mx + EB*mu
	Φ = mW

	# Set outgoing message
	return Message(Univariate, GaussianMeanPrecision, m=ϕ, w=Φ)
end

function ruleVariationalGFXIn1PNPPPP(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Multivariate},
                                     marg_θ :: Nothing,
									 marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_η :: ProbabilityDistribution{Univariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mx,Vx = unsafeMeanCov(marg_x)
	mη,Vη = unsafeMeanCov(marg_η)
	mu,Vu = unsafeMeanCov(marg_u)
	mτ,Vτ = unsafeMeanCov(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = 1.

	# Cast precision to matrix
	mW = wMatrix(mτ, order)

	# Set parameters
	ϕ = mx*s'*mW*(my - s*mη*mu) - mW*S'*Vx*s
	Φ = mτ*(mx*mx' + Vx)

	# Set outgoing message
	return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalGFXIn2PPNPPP(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Multivariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: Nothing,
									 marg_η :: ProbabilityDistribution{Univariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

   	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mη,Vη = unsafeMeanCov(marg_η)
	mu,Vu = unsafeMeanCov(marg_u)
	mτ,Vτ = unsafeMeanCov(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = 1.

	# Cast precision to matrix
	mW = wMatrix(mτ, order)
	
	# Compute expected values
	EA = S + s*mθ'

	# Set parameters
	ϕ = EA*mW*(my - s*mη*mu)
	Φ = mW*(EA'*EA + Vθ)

	# Set outgoing message
	return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalGFXIn3PPPNPP(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Multivariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_η :: Nothing,
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mx,Vx = unsafeMeanCov(marg_x)
	mu,Vu = unsafeMeanCov(marg_u)
	mτ,Vτ = unsafeMeanCov(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = 1.

	# Cast precision to matrix
	mW = wMatrix(mτ, order)
	
	# Compute expected values
	EA = S + s*mθ'

	# Set parameters
	ϕ = mu'*s'*mW*(my - EA*mx)
	Φ = mτ*(mu*mu' + Vu)

	# Set outgoing message
	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalGFXIn4PPPPNP(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Multivariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
									 marg_η :: ProbabilityDistribution{Univariate},
								     marg_u :: Nothing,
                                     marg_τ :: ProbabilityDistribution{Univariate})

   
	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mx,Vx = unsafeMeanCov(marg_x)
	mη,Vη = unsafeMeanCov(marg_η)
	mτ,Vτ = unsafeMeanCov(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = 1.

	# Cast precision to matrix
	mW = wMatrix(mτ, order)
	
	# Compute expected values
	EA = S + s*mθ'

	# Set parameters
	ϕ = mη'*s'*mW*(my - EA*mx)
	Φ = mτ*(mη*mη' + Vη)

	# Set outgoing message
	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalGFXIn5PPPPPN(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Multivariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
									 marg_η :: ProbabilityDistribution{Univariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: Nothing)

   
	# Extract moments of beliefs
	my,Vy = unsafeMeanCov(marg_y)
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mx,Vx = unsafeMeanCov(marg_x)
	mη,Vη = unsafeMeanCov(marg_η)
	mu,Vu = unsafeMeanCov(marg_u)
	
	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = 1.

	# Cast precision to matrix
	mW = wMatrix(mτ, order)
	
	# Compute expected values
	EA = S + s*mθ'
	EB = s*mη

	# Set parameters
	a = 3/2
	b = my[order] + Vy[order,order] -2*my[order]*(EA*mx)[order] -2*(EA*mx)[order]*(EB*mu)[order] -2*my[order]*(EB*mu)[order] + mx'*(EA'*EA + Vθ)*mx + tr((EA'*EA + Vθ)*Vx) + (mη^2 + Vη)*(mu^2 + Vu)

	# Set outgoing message
	return Message(Univariate, Gamma, a=a, b=b)
end

function collectNaiveVariationalNodeInbounds(node::GeneralisedFilterX, entry::ScheduleEntry)
	inbounds = Any[]

	# Push function to calling signature (Δt needs to be defined in user scope)
	push!(inbounds, Dict{Symbol, Any}(:Δt => node.Δt, :keyword => false))

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
