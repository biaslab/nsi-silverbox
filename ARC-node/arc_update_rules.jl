import LinearAlgebra: I, Hermitian, tr
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType
include("util.jl")

export ruleVariationalARCOutNPPPPP,
       ruleVariationalARCIn1PNPPPP,
       ruleVariationalARCIn2PPNPPP,
       ruleVariationalARCIn3PPPNPP,
	   ruleVariationalARCIn4PPPPNP,
	   ruleVariationalARCIn5PPPPPN
       # ruleSVariationalARCOutNPPP,
       # ruleSVariationalARCIn1PNPP,
       # ruleSVariationalARCIn2PPNP,
       # ruleSVariationalARCIn3PPPN,
       # ruleMGaussianMeanVarianceGGGD

global order = Nothing
global c = Nothing
global S = Nothing

function defineOrder(dim)
    global order, c, S
    order = dim
    c = uvector(order)
    S = shift(order)
end

function ruleVariationalARCOutNPPPPP(marg_y :: Nothing,
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
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                       defineOrder(length(mθ)) : order

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

    # Parameters of outgoing message
    z = mW*((S+c*mθ')*mx + c*mη*mu)
	D = mW
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalARCIn1PNPPPP(marg_y :: ProbabilityDistribution{Multivariate},
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
    order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
                     defineOrder(length(mθ)) : order

    # Expectation of AR coefficient matrix
    mA = S + c*mθ'

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

    # Parameters of outgoing message
    D = mA'*mW*mA + Vθ*mγ
    z = mA'*mW*(my - c*mη*mu)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalARCIn2PPNPPP(marg_y :: ProbabilityDistribution{Multivariate},
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

    # Parameters of outgoing message
    D = mγ*(Vx + mx*mx')
    z = mx*c'*mW*(my - c*mη*mu)
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalARCIn3PPPNPP(marg_y :: ProbabilityDistribution{Multivariate},
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

	# Check order
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

    # Expectation of AR coefficient matrix
    mA = S + c*mθ'

	# Parameters of outgoing message
	D = mγ*(mu^2 + vu)
    z = (mu*c')*mW*(my - mA*mx)
    Message(Univariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalARCIn4PPPPNP(marg_y :: ProbabilityDistribution{Multivariate},
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

	# Check order
    order == Nothing ? defineOrder(length(my)) : order != length(my) ?
                       defineOrder(length(my)) : order

    # Map transition noise to matrix
    mW = wMatrix(mγ, order)

    # Expectation of AR coefficient matrix
    mA = S + c*mθ'

	# Parameters of outgoing message
	D = mγ*(mη^2 + vη)
    z = (mη*c')*mW*(my - mA*mx)
    Message(Univariate, GaussianWeightedMeanPrecision, xi=z, w=D)
end

function ruleVariationalARCIn5PPPPPN(marg_y :: ProbabilityDistribution{Multivariate},
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

	# Parameters of outgoing message
	a = 3/2
    B = Vy[1,1] + my[1]*my[1] -2*my[1]*(mθ'*mx + mη*mu) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ + 2*mθ'*mx*mη*mu + (mu^2+vu)*(mη^2+vη)
    Message(Gamma, a=a, b=B/2)
end

# function ruleVariationalARCIn2PPNP(marg_y :: ProbabilityDistribution{V, PointMass},
#                                    marg_x :: ProbabilityDistribution{Multivariate, PointMass},
#                                    marg_θ :: Nothing,
#                                    marg_γ :: ProbabilityDistribution{Univariate}) where V<:VariateType
#     my = unsafeMean(marg_y)
#     order == Nothing ? defineOrder(length(my)) : order != length(my) ?
#                        defineOrder(length(my)) : order
#     mx = unsafeMean(marg_x)
#     mγ = unsafeMean(marg_γ)
#     W =  mx*mγ*mx'
#     xi = (mx*c'*wMatrix(mγ, order)*my)
#     Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
# end
#
# function ruleVariationalARCIn3PPPN(marg_y :: ProbabilityDistribution{V, PointMass},
#                                    marg_x :: ProbabilityDistribution{Multivariate, PointMass},
#                                    marg_θ :: ProbabilityDistribution{Multivariate},
#                                    marg_γ :: Nothing) where V<:VariateType
#     mθ = unsafeMean(marg_θ)
#     my = unsafeMean(marg_y)
#     mx = unsafeMean(marg_x)
#     Vθ = unsafeCov(marg_θ)
#     B = my[1]*my[1] - 2*my[1]*mθ'*mx + mx'*(Vθ+mθ'mθ)*mx
#     Message(Gamma, a=3/2, b=B/2)
# end
#
# # No copying behavior (classical AR)
#
# function ruleVariationalARCIn2PPNP(marg_y :: ProbabilityDistribution{Univariate, PointMass},
#                                    marg_x :: ProbabilityDistribution{Multivariate, PointMass},
#                                    marg_θ :: Nothing,
#                                    marg_γ :: ProbabilityDistribution{Univariate})
#     my = unsafeMean(marg_y)
#     order == Nothing ? defineOrder(length(my)) : order != length(my) ?
#                        defineOrder(length(my)) : order
#     mx = unsafeMean(marg_x)
#     mγ = unsafeMean(marg_γ)
#     W =  mγ*mx*mx'
#     xi = (mγ*mx*my)
#     Message(Multivariate, GaussianWeightedMeanPrecision, xi=xi, w=W)
# end
#
# function ruleVariationalARCIn3PPPN(marg_y :: ProbabilityDistribution{Univariate, PointMass},
#                                    marg_x :: ProbabilityDistribution{Multivariate, PointMass},
#                                    marg_θ :: ProbabilityDistribution{Multivariate},
#                                    marg_γ :: Nothing)
#     mθ = unsafeMean(marg_θ)
#     my = unsafeMean(marg_y)
#     mx = unsafeMean(marg_x)
#     Vθ = unsafeCov(marg_θ)
#     B = my[1]*my[1] - 2*my[1]*mθ'*mx + mx'*Vθ*mx + mθ'*mx*mx'*mθ
#     Message(Gamma, a=3/2, b=B/2)
# end
#
# function ruleMGaussianMeanVarianceGGGD(msg_y::Message{F1, V},
#                                        msg_x::Message{F2, V},
#                                        dist_θ::ProbabilityDistribution,
#                                        dist_γ::ProbabilityDistribution) where {F1<:Gaussian, F2<:Gaussian, V<:VariateType}
#
#     mθ = unsafeMean(dist_θ)
#     mA = S+c*mθ'
#     mγ = unsafeMean(dist_γ)
#     Vθ = unsafeCov(dist_θ)
#     order == Nothing ? defineOrder(length(mθ)) : order != length(mθ) ?
#                        defineOrder(length(mθ)) : order
#     trans = transition(mγ, order)
#     mW = wMatrix(mγ, order)
#
#     b_my = unsafeMean(msg_y.dist)
#     b_Vy = unsafeCov(msg_y.dist)
#     f_mx = unsafeMean(msg_x.dist)
#     f_Vx = unsafeCov(msg_x.dist)
#     D = inv(f_Vx) + mγ*Vθ
#     W = [inv(b_Vy)+mW -mW*mA; -mA'*mW D+mA'*mW*mA]
#     m = inv(W)*[inv(b_Vy)*b_my; inv(f_Vx)*f_mx]
#     return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=m, v=inv(W))
# end
