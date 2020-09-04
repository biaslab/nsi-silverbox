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
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mη = unsafeMean(marg_η)
	mu = unsafeMean(marg_u)
	mτ = unsafeMean(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = 1.

	# Cast precision to matrix
	mW = wMatrix(mτ, order)

	# Compute expected values
	EA = (S + s*mθ')
	EB = s*mη

	# Set parameters
	ϕ = EA*mx + EB*mu
	Φ = mW

	# Set outgoing message
	return Message(Univariate, GaussianMeanPrecision, m=ϕ, w=Φ)
end

function ruleVariationalGFXIn1PNPPPP(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Univariate},
                                     marg_θ :: Nothing,
									 marg_x :: ProbabilityDistribution{Multivariate},
                                     marg_z :: ProbabilityDistribution{Multivariate},
                                     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})


	# Extract moments of beliefs
	mθ,Vθ = unsafeMeanCov(marg_θ)
	mx,Vx = unsafeMeanCov(marg_x)
	mη = unsafeMean(marg_η)
	mu = unsafeMean(marg_u)
	mτ = unsafeMean(marg_τ)

	# Set order
	order = dims(marg_θ)

	# Define helper matrices
	S = Bidiagonal(ones(order,), Δt.*ones(order-1,), :U)
	s = zeros(order,); s[end] = 1.

	# Cast precision to matrix
	mW = wMatrix(mτ, order)


end

function ruleVariationalGFXIn2PPNPPP(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: Nothing,
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

   #TODO
end

function ruleVariationalGFXIn3PPPNPP(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: Nothing,
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: ProbabilityDistribution{Univariate})

   #TODO
end

function ruleVariationalGFXIn4PPPPNP(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: Nothing,
                                     marg_τ :: ProbabilityDistribution{Univariate})

   #TODO
end

function ruleVariationalGFXIn5PPPPPN(Δt :: Float,
									 marg_y :: ProbabilityDistribution{Univariate},
									 marg_θ :: ProbabilityDistribution{Multivariate},
									 marg_x :: ProbabilityDistribution{Multivariate},
							  	     marg_z :: ProbabilityDistribution{Multivariate},
								     marg_u :: ProbabilityDistribution{Univariate},
                                     marg_τ :: Nothing)

   #TODO
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
