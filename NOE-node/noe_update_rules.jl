import LinearAlgebra: I, Hermitian, tr, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType
include("util.jl")

export ruleVariationalNOEOutNPPPPP,
       ruleVariationalNOEIn1PNPPPP,
       ruleVariationalNOEIn2PPNPPP,
       ruleVariationalNOEIn3PPPNPP,
	   ruleVariationalNOEIn4PPPPNP,
	   ruleVariationalNOEIn5PPPPPN


function ruleVariationalNOEOutNPPPPP(marg_y :: Nothing,
									 marg_θ :: ProbabilityDistribution{Multivariate},
                                     marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_z :: ProbabilityDistribution{Multivariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

	#TODO
end

function ruleVariationalNOEIn1PNPPPP(marg_y :: ProbabilityDistribution{Univariate},
                                     marg_θ :: Nothing,
									 marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_z :: ProbabilityDistribution{Multivariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

    #TODO
end

function ruleVariationalNOEIn2PPNPPP(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: Nothing,
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

    #TODO
end

function ruleVariationalNOEIn3PPPNPP(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: Nothing,
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

    #TODO
end

function ruleVariationalNOEIn4PPPPNP(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: Nothing,
                                     marg_τ :: ProbabilityDistribution{Univariate})

    #TODO
end

function ruleVariationalNOEIn5PPPPPN(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: Nothing)

    #TODO
end
