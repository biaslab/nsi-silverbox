import LinearAlgebra: I, Hermitian, tr, pinv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType,
				  collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner
import Zygote: gradient

export ruleVariationalNLARXOutNPPPPP,
       ruleVariationalNLARXIn1PNPPPP,
       ruleVariationalNLARXIn2PPNPPP,
       ruleVariationalNLARXIn3PPPNPP,
	   ruleVariationalNLARXIn4PPPPNP,
	   ruleVariationalNLARXIn5PPPPPN

# Autoregression order and bookkeeping matrices
order = Nothing
S = Nothing
s = Nothing

# Approximating points
approxx = 0.0
approxθ = 0.0

function defineOrder(dim::Int64)
    global order, s, S

	# Set autoregression order
    order = dim

	# Set bookkeeping matrices
    s = uvector(order)
    S = shift(order)
end


function ruleVariationalNLARXOutNPPPPP(g :: Function,
									   marg_y :: Nothing,
                                       marg_x :: ProbabilityDistribution{Multivariate},
                                       marg_θ :: ProbabilityDistribution{Multivariate},
                                       marg_η :: ProbabilityDistribution{Univariate},
                                       marg_u :: ProbabilityDistribution{Univariate},
                                       marg_γ :: ProbabilityDistribution{Univariate})

    # Expectations of incoming marginal beliefs
    mx = unsafeMean(marg_x)
    mθ = unsafeMean(marg_θ)
    mη = unsafeMean(marg_η)
    mu = unsafeMean(marg_u)
    mγ = unsafeMean(marg_γ)

    # Check order
	if order == Nothing
		defineOrder(length(mx))
	end

	# Update approximating points
	global approxx = mx
	global approxθ = mθ

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

    # Parameters of outgoing message
	Φ = mW
    ϕ = mW*(S*mx + s*g(mθ, mx) + s*mη*mu)

	# Set outgoing message
	return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNLARXIn1PNPPPP(g :: Function,
									   marg_y :: ProbabilityDistribution{Multivariate},
                                       marg_x :: Nothing,
                                       marg_θ :: ProbabilityDistribution{Multivariate},
                                       marg_η :: ProbabilityDistribution{Univariate},
                                       marg_u :: ProbabilityDistribution{Univariate},
                                       marg_γ :: ProbabilityDistribution{Univariate})

    # Expectations of marginal beliefs
    my = unsafeMean(marg_y)
    mθ = unsafeMean(marg_θ)
    Vθ = unsafeCov(marg_θ)
    mη = unsafeMean(marg_η)
    mu = unsafeMean(marg_u)
    mγ = unsafeMean(marg_γ)

	# Check order
	if order == Nothing
		defineOrder(length(mx))
	end

	# Check approximating points
	if approxx == Nothing
		error("Approximating point for x not initialized")
	end

	# Gradient of supplied nonlinear function
	Jθ, Jx = gradient(g, mθ, approxx)

	# Map transition noise to matrix
	mW = wMatrix(mγ, order)

    # Parameters of outgoing message
    Φ = (S + s*Jx')'*mW*(S + s*Jx')
    ϕ = (S + s*Jx')'*mW*(my - s*mη*mu)

	# Update global approximating point
	global approxx = inv(Φ + 1e-6*Matrix{Float64}(I, size(Φ)))*ϕ
	global approxθ = mθ

    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNLARXIn2PPNPPP(g :: Function,
									   marg_y :: ProbabilityDistribution{Multivariate},
                                       marg_x :: ProbabilityDistribution{Multivariate},
                                       marg_θ :: Nothing,
							  	       marg_η :: ProbabilityDistribution{Univariate},
								       marg_u :: ProbabilityDistribution{Univariate},
                                       marg_γ :: ProbabilityDistribution{Univariate})


	# Expectations of marginal beliefs
	my = unsafeMean(marg_y)
	mx = unsafeMean(marg_x)
	Vx = unsafeCov(marg_x)
	mη = unsafeMean(marg_η)
	mu = unsafeMean(marg_u)
	mγ = unsafeMean(marg_γ)

	# Check order
	if order == Nothing
		defineOrder(length(mx))
	end

	# Check approximating points
	if approxθ == Nothing
		error("Approximating point for θ not initialized")
	end

	# Gradient of supplied nonlinear function
	Jθ, Jx = gradient(g, approxθ, mx)

	# Map transition noise to matrix
	mW = wMatrix(mγ, order)

    # Parameters of outgoing message
    Φ = mγ*Jθ*Jθ'
    ϕ = Jθ*s'*mW*(my - s*mη*mu - s*(g(approxθ, mx) - Jθ'*approxθ))

	# Update global approximating point
	global approxx = mx
	global approxθ = inv(Φ + 1e-6*Matrix{Float64}(I, size(Φ)))*ϕ

    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNLARXIn3PPPNPP(g :: Function,
									   marg_y :: ProbabilityDistribution{Multivariate},
                                       marg_x :: ProbabilityDistribution{Multivariate},
                                       marg_θ :: ProbabilityDistribution{Multivariate},
								       marg_η :: Nothing,
								       marg_u :: ProbabilityDistribution{Univariate},
                                       marg_γ :: ProbabilityDistribution{Univariate})

 	# Expectations of marginal beliefs
	mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
	mu = unsafeMean(marg_u)
	vu = unsafeCov(marg_u)
	mγ = unsafeMean(marg_γ)

	# Update global approximating point
	global approxx = mx
	global approxθ = mθ

	# Check order
	if order == Nothing
		defineOrder(length(mx))
	end

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

	# Parameters of outgoing message
	Φ = mγ*(mu^2 + vu)
    ϕ = (mu*s')*mW*(my - (S*mx + s*g(mθ, mx)))

	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNLARXIn4PPPPNP(g :: Function,
									   marg_y :: ProbabilityDistribution{Multivariate},
                                       marg_x :: ProbabilityDistribution{Multivariate},
                                       marg_θ :: ProbabilityDistribution{Multivariate},
							  	       marg_η :: ProbabilityDistribution{Univariate},
								       marg_u :: Nothing,
                                       marg_γ :: ProbabilityDistribution{Univariate})

 	# Expectations of marginal beliefs
	mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
	mη = unsafeMean(marg_η)
	vη = unsafeCov(marg_η)
	mγ = unsafeMean(marg_γ)

	# Update global approximating point
	global approxx = mx
	global approxθ = mθ

	# Check order
	if order == Nothing
		defineOrder(length(mx))
	end

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

	# Parameters of outgoing message
	Φ = mγ*(mη^2 + vη)
    ϕ = (mη*s')*mW*(my - (S*mx + s*g(mθ, mx)))

	return Message(Univariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNLARXIn5PPPPPN(g :: Function,
									   marg_y :: ProbabilityDistribution{Multivariate},
                                       marg_x :: ProbabilityDistribution{Multivariate},
                                       marg_θ :: ProbabilityDistribution{Multivariate},
							  	       marg_η :: ProbabilityDistribution{Univariate},
								       marg_u :: ProbabilityDistribution{Univariate},
                                       marg_γ :: Nothing)

    # Expectations of marginal beliefs
	mθ = unsafeMean(marg_θ)
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
	mη = unsafeMean(marg_η)
	mu = unsafeMean(marg_u)
    Vθ = unsafeCov(marg_θ)
    Vy = unsafeCov(marg_y)
    Vx = unsafeCov(marg_x)
	vη = unsafeCov(marg_η)
	vu = unsafeCov(marg_u)

	# Update global approximating point
	global approxx = mx
	global approxθ = mθ

	# Check order
	if order == Nothing
		defineOrder(length(mx))
	end

	# Gradient of supplied nonlinear function
	Jθ, Jx = gradient(g, mθ, approxx)

	# Convenience variables
	Aθx = S*mx + s*g(mθ, mx)

	# Intermediate terms
	term1 = (my*my' + Vy)[1,1]
	term2 = -(Aθx*my')[1,1]
	term3 = -((s*mη*mu)*my')[1,1]
	term4 = -(my*Aθx')[1,1]
	term5 = (mx'*S'*S*mx)[1,1] + (S*Vx*S')[1,1] + g(mθ, mx)^2 + Jx'*Vx*Jx + Jθ'*Vθ*Jθ
	term6 = ((s*mη*mu)*Aθx')[1,1]
	term7 = -(my*(s*mη*mu)')[1,1]
	term8 = (Aθx*(s*mη*mu)')[1,1]
	term9 = (mu^2 + vu)*(mη^2 + vη)

	# Parameters of outgoing message
	a = 3/2
    B = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)/2

	return Message(Gamma, a=a, b=B)
end


function collectNaiveVariationalNodeInbounds(node::NLatentAutoregressiveX, entry::ScheduleEntry)
	inbounds = Any[]

	# Push function to calling signature (g needs to be defined in user scope)
	push!(inbounds, Dict{Symbol, Any}(:g => node.g, :keyword => false))

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
