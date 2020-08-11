import LinearAlgebra: I, Hermitian, tr
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType
import Zygote: gradient
include("util.jl")

export ruleVariationalGFXOutNPPPPP,
       ruleVariationalGFXIn1PNPPPP,
       ruleVariationalGFXIn2PPNPPP,
       ruleVariationalGFXIn3PPPNPP,
	   ruleVariationalGFXIn4PPPPNP,
	   ruleVariationalGFXIn5PPPPPN


function ruleVariationalGFXOutNPPPPP(marg_y :: Nothing,
									 marg_θ :: ProbabilityDistribution{Multivariate},
                                     marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_z :: ProbabilityDistribution{Multivariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

	#TODO
end

function ruleVariationalGFXIn1PNPPPP(marg_y :: ProbabilityDistribution{Univariate},
                                     marg_θ :: Nothing,
									 marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_z :: ProbabilityDistribution{Multivariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

   #TODO
end

function ruleVariationalGFXIn2PPNPPP(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: Nothing,
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

   #TODO
end

function ruleVariationalGFXIn3PPPNPP(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: Nothing,
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

   #TODO
end

function ruleVariationalGFXIn4PPPPNP(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: Nothing,
                                     marg_τ :: ProbabilityDistribution{Univariate})

   #TODO
end

function ruleVariationalGFXIn5PPPPPN(marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: Nothing)

   #TODO
end