import LinearAlgebra: I, Hermitian, tr
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType
# import Zygote: gradient
include("util.jl")

export ruleVariationalNLARXOutNPPPPP,
       ruleVariationalNLARXIn1PNPPPP,
       ruleVariationalNLARXIn2PPNPPP,
       ruleVariationalNLARXIn3PPPNPP,
	   ruleVariationalNLARXIn4PPPPNP,
	   ruleVariationalNLARXIn5PPPPPN

order = Nothing
c = Nothing
S = Nothing
mx_ = zeros(2,)
mθ_ = zeros(3,)

function defineOrder(dim::Int64)
    global order, c, S
    order = dim
    c = uvector(order)
    S = shift(order)
end

function g(x::Array{Float64,1}, θ::Array{Float64,1})
	"Hard-coded nonlinearity (todo: g as an input argument to rule)"
	return θ[1]*x[1] + θ[2]*x[1]^3 + θ[3]*x[2]
end

function hardcoded_gradient(mx::Array{Float64,1}, mθ::Array{Float64,1})
	Jx = Array([mθ[1] + mθ[2]*3*mx[1]^2, mθ[3]])
	Jθ = Array([mx[1], mx[1]^3, mx[2]])
	return Jx, Jθ
end

function ruleVariationalNLARXOutNPPPPP(marg_y :: Nothing,
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

	# Update global approximating point
	global mx_ = mx
	global mθ_ = mθ

    # Check order
    order == Nothing ? defineOrder(length(mx)) : order != length(mx) ?
                       defineOrder(length(mx)) : order

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

    # Parameters of outgoing message
    z = mW*(S*mx + c*g(mx, mθ) + c*mη*mu)
	D = mW

	return Message(Multivariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalNLARXIn1PNPPPP(marg_y :: ProbabilityDistribution{Multivariate},
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
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

	# Jacobians of nonlinearity
	# Jx, Jθ = gradient(g, mx_, mθ)
	Jx, Jθ = hardcoded_gradient(mx_, mθ)

    # Parameters of outgoing message
    D = (S + c*Jx')'*mW*(S + c*Jx')
    z = (S + c*Jx')'*mW*(my - c*mη*mu)

	# Update global approximating point
	global mx_ = cholinv(D)*z
	global mθ_ = mθ

    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalNLARXIn2PPNPPP(marg_y :: ProbabilityDistribution{Multivariate},
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
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

	# Jacobians of nonlinearity
	# Jx, Jθ = gradient(g, mx, mθ_)
	Jx, Jθ = hardcoded_gradient(mx, mθ_)

    # Parameters of outgoing message
    D = mγ*(Jθ*Jθ')
    z = Jθ*c'*mW*(my - c*mη*mu - c*(g(mx,mθ_) - Jθ'*mθ_))

	# Update global approximating point
	global mx_ = mx
	global mθ_ = cholinv(D)*z

    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalNLARXIn3PPPNPP(marg_y :: ProbabilityDistribution{Multivariate},
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
	global mx_ = mx
	global mθ_ = mθ

	# Check order
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

	# Parameters of outgoing message
	D = mγ*(mu^2 + vu)
    z = (mu*c')*mW*(my - (S*mx + c*g(mx,mθ)))

	return Message(Univariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalNLARXIn4PPPPNP(marg_y :: ProbabilityDistribution{Multivariate},
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
	global mx_ = mx
	global mθ_ = mθ

	# Check order
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

	# Parameters of outgoing message
	D = mγ*(mη^2 + vη)
    z = (mη*c')*mW*(my - (S*mx + c*g(mx,mθ)))

	return Message(Univariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalNLARXIn5PPPPPN(marg_y :: ProbabilityDistribution{Multivariate},
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
	global mx_ = mx
	global mθ_ = mθ

	# Jacobians of nonlinearity
	# Jx, Jθ = gradient(g, mx, mθ)
	Jx, Jθ = hardcoded_gradient(mx, mθ)

	# Convenience variables
	Aθx = S*mx + c*g(mx,mθ)

	# Intermediate terms
	tmp1 = (my*my' + Vy)[1,1]
	tmp2 = -(Aθx*my')[1,1]
	tmp3 = -((c*mη*mu)*my')[1,1]
	tmp4 = -(my*Aθx')[1,1]
	tmp5 = (mx'*S'*S*mx)[1,1] + (S*Vx*S')[1,1] + g(mx,mθ)^2 + Jx'*Vx*Jx + Jθ'*Vθ*Jθ
	tmp6 = ((c*mη*mu)*Aθx')[1,1]
	tmp7 = -(my*(c*mη*mu)')[1,1]
	tmp8 = (Aθx*(c*mη*mu)')[1,1]
	tmp9 = (mu^2 + vu)*(mη^2 + vη)

	# Parameters of outgoing message
	a = 3/2
    B = tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 + tmp8 + tmp9

	return Message(Gamma, a=a, b=B/2)
end
